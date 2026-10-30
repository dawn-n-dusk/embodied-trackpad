import Foundation

// Device model: derives relative cursor motion, force-hysteresis click state and
// two-finger centroid scrolling from touch.v1 contacts. It never sees screen
// coordinates, window names, pids, or target coordinates; input is pad-relative
// meters and output is clamped inside the content area.
struct PadConfig {
    var padWidthM: Double = 0.16
    var padHeightM: Double = 0.11
    var gainPxPerM: Double = 4000
    var pressForceN: Double = 1.0
    var releaseForceN: Double = 0.6
    var idleTimeoutS: Double = 0.5 // simulation-time inactivity
    var wallIdleTimeoutS: Double = 0.8 // host monotonic stalled-transport watchdog
    var maxContacts = 10
    var maxDeltaM: Double = 0.05
    var contentWidthPx: Double = 800
    var contentHeightPx: Double = 600
    var halfWidthM: Double { padWidthM / 2 }
    var halfHeightM: Double { padHeightM / 2 }
}

struct Contact {
    var fingerId: String
    var xM: Double
    var yM: Double
    var forceN: Double
}

enum TouchRejection: Error {
    case unknownField(String)
    case wrongSchema(String)
    case badType(String)
    case nonFinite(String)
    case outOfOrderSequence(expected: Int, got: Int)
    case nonMonotonicTime(last: Double, got: Double)
    case tooManyContacts(Int)
    case duplicateFingerId(String)
    case emptyFingerId
    case contactOutOfRange(String)
    case excessiveDelta(String)
}

// Gesture events surfaced to the app layer for task bookkeeping.
struct GestureEvents {
    var pressBegan = false
    var releaseAt: (x: Double, y: Double)? = nil // button released (a click)
}

final class TouchProcessor {
    var config: PadConfig
    private(set) var cursorXPx: Double
    private(set) var cursorYPx: Double
    private(set) var buttonDown = false
    private(set) var scrollOffsetPx: Double = 0
    private(set) var pressSuppressed = false
    private var origins: [String: (x: Double, y: Double)] = [:]
    private var lastMessageTimeS: Double?
    private var nextSequence = 1

    init(config: PadConfig) {
        self.config = config
        self.cursorXPx = config.contentWidthPx / 2
        self.cursorYPx = config.contentHeightPx / 2
    }

    func resetCursor() {
        cursorXPx = config.contentWidthPx / 2
        cursorYPx = config.contentHeightPx / 2
        buttonDown = false
        scrollOffsetPx = 0
        pressSuppressed = false
        origins.removeAll()
    }

    // Hard cancel of any in-flight gesture: no click is emitted, drag arming is
    // dropped by the app layer, and pressing stays suppressed until a fresh
    // lift (an empty-contact frame) is observed.
    func cancelGesture() {
        buttonDown = false
        origins.removeAll()
        pressSuppressed = true
    }

    func releaseAll() { // stop/EOF: full gesture-state reset
        cancelGesture()
    }

    // Wall-clock watchdog: a stalled transport must release the held button.
    // Host monotonic time is independent of the simulation clock.
    func wallClockIdleCancel(idleS: Double) -> Bool {
        guard buttonDown || !origins.isEmpty else { return false }
        guard idleS > config.wallIdleTimeoutS else { return false }
        cancelGesture()
        return true
    }

    // Numeric fields must be real JSON numbers. NSNumber bridges JSON
    // booleans to 1/0 (and `is Bool`/`as? Bool` cannot tell NSNumber(1) from
    // true), so booleans are identified by their CoreFoundation type ID.
    static func isBoolValue(_ v: Any) -> Bool {
        guard let n = v as? NSNumber else { return false }
        return CFGetTypeID(n) == CFBooleanGetTypeID()
    }

    static func strictDouble(_ v: Any?, _ field: String) throws -> Double {
        guard let v = v, !isBoolValue(v), let d = v as? Double, d.isFinite else {
            throw TouchRejection.nonFinite(field)
        }
        return d
    }

    static func strictInt(_ v: Any?, _ field: String) throws -> Int {
        guard let v = v, !isBoolValue(v), let i = v as? Int else {
            throw TouchRejection.badType("\(field) must be int")
        }
        return i
    }

    // Validates raw message fields, then applies the gesture. All validation
    // happens before any state is consumed, so a rejected message never
    // advances sequence, time, or origins.
    func handleTouch(schemaVersion: Any?, sequence: Any?, timeS: Any?, rawContacts: Any?) throws -> GestureEvents {
        var events = GestureEvents()
        guard let schema = schemaVersion as? String, schema == "touch.v1" else {
            throw TouchRejection.wrongSchema("schema_version must be 'touch.v1'")
        }
        let seq = try TouchProcessor.strictInt(sequence, "sequence")
        guard seq == nextSequence else { throw TouchRejection.outOfOrderSequence(expected: nextSequence, got: seq) }
        let t = try TouchProcessor.strictDouble(timeS, "simulation_time_s")
        guard t >= 0 else { throw TouchRejection.nonFinite("simulation_time_s") }
        if let last = lastMessageTimeS, t < last { throw TouchRejection.nonMonotonicTime(last: last, got: t) }
        guard let rawList = rawContacts as? [[String: Any]] else { throw TouchRejection.badType("contacts must be an array of objects") }
        guard rawList.count <= config.maxContacts else { throw TouchRejection.tooManyContacts(rawList.count) }

        var contacts: [Contact] = []
        var seen: Set<String> = []
        for (i, rc) in rawList.enumerated() {
            for key in rc.keys where !["finger_id", "x_m", "y_m", "normal_force_n"].contains(key) {
                throw TouchRejection.unknownField("contacts[\(i)].\(key)")
            }
            guard let fid = rc["finger_id"] as? String else { throw TouchRejection.badType("finger_id") }
            guard !fid.isEmpty else { throw TouchRejection.emptyFingerId }
            guard seen.insert(fid).inserted else { throw TouchRejection.duplicateFingerId(fid) }
            let x = try TouchProcessor.strictDouble(rc["x_m"], "x_m")
            let y = try TouchProcessor.strictDouble(rc["y_m"], "y_m")
            let f = try TouchProcessor.strictDouble(rc["normal_force_n"], "normal_force_n")
            guard f >= 0 else { throw TouchRejection.nonFinite("normal_force_n") }
            // x/y are pad-centered: only half the pad extends to each side.
            guard abs(x) <= config.halfWidthM, abs(y) <= config.halfHeightM else {
                throw TouchRejection.contactOutOfRange(fid)
            }
            contacts.append(Contact(fingerId: fid, xM: x, yM: y, forceN: f))
        }

        // Excessive per-sample motion for fingers matched across frames is
        // rejected, not clamped.
        for c in contacts {
            if let o = origins[c.fingerId] {
                let dx = c.xM - o.x, dy = c.yM - o.y
                if abs(dx) > config.maxDeltaM || abs(dy) > config.maxDeltaM {
                    throw TouchRejection.excessiveDelta(c.fingerId)
                }
            }
        }

        nextSequence += 1

        // Simulation-time inactivity expiry applies BEFORE any gesture
        // evaluation, including a lift: a press followed by a long sim-time
        // gap is cancelled silently and must not emit a release/click.
        if let last = lastMessageTimeS, t - last > config.idleTimeoutS {
            cancelGesture()
        }
        lastMessageTimeS = t

        if contacts.isEmpty {
            // Lift: releases a held button as a click; clears suppression so a
            // fresh gesture can start (also after a cancellation).
            origins.removeAll()
            pressSuppressed = false
            if buttonDown {
                buttonDown = false
                events.releaseAt = (cursorXPx, cursorYPx)
            }
            return events
        }

        // Finger-set / gesture-mode change away from an active one-finger
        // press cancels it silently (no fake click) and suppresses re-press.
        if buttonDown && !(contacts.count == 1 && origins[contacts[0].fingerId] != nil) {
            buttonDown = false
            pressSuppressed = true
        }

        if contacts.count == 1 {
            let c = contacts[0]
            if let o = origins[c.fingerId] {
                // First contact has no origin: no cursor jump (no teleport).
                cursorXPx = min(config.contentWidthPx, max(0, cursorXPx + (c.xM - o.x) * config.gainPxPerM))
                cursorYPx = min(config.contentHeightPx, max(0, cursorYPx + (c.yM - o.y) * config.gainPxPerM))
            }
            if !buttonDown && !pressSuppressed && c.forceN >= config.pressForceN {
                buttonDown = true
                events.pressBegan = true
            } else if buttonDown && c.forceN <= config.releaseForceN {
                buttonDown = false
                events.releaseAt = (cursorXPx, cursorYPx)
            }
        } else if contacts.count == 2 {
            // Two-finger centroid scrolling only; no press state.
            if let o0 = origins[contacts[0].fingerId], let o1 = origins[contacts[1].fingerId] {
                let cy = (contacts[0].yM + contacts[1].yM) / 2
                let ocy = (o0.y + o1.y) / 2
                scrollOffsetPx = max(0, scrollOffsetPx + (cy - ocy) * config.gainPxPerM)
            }
        }
        // 3+ contacts: origins tracked only, no gesture.

        // A finger omitted from this frame loses its origin even if other
        // fingers remain; a returning finger starts fresh and cannot jump.
        var newOrigins: [String: (x: Double, y: Double)] = [:]
        for c in contacts { newOrigins[c.fingerId] = (c.xM, c.yM) }
        origins = newOrigins
        return events
    }
}
