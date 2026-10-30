import AppKit
import Foundation

// TrackpadSandbox: an original, bounded native test GUI for embodied-trackpad.
// It is NOT Finder, NOT whole-macOS, and NOT a VM. It renders its own task
// surface (virtual cursor, click target, drag target/destination, scroll
// content), accepts touch.v1 JSONL contacts only on inherited stdin, and emits
// JSONL responses plus base64 PNGs of its own content view on stdout.
// It performs no CGEvent posting, no system-pointer moves, no AppleScript,
// no AX automation, no network, and no file access outside its container.

let CONTENT_W: CGFloat = 800
let CONTENT_H: CGFloat = 600

// Deterministic layout from a seed (LCG), so reset(seed) is reproducible.
struct Layout {
    var clickTarget: CGRect
    var dragSource: CGRect
    var dragDestination: CGRect
    var scrollMaxPx: CGFloat
}

func layoutForSeed(_ seed: UInt64) -> Layout {
    var s = seed &* 6364136223846793005 &+ 1442695040888963407
    func next() -> CGFloat {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        return CGFloat((s >> 33) & 0xFFFFFF) / CGFloat(0x1000000)
    }
    let tx = 40 + next() * 200
    let ty = 330 + next() * 200
    let sx = 40 + next() * 150
    let sy = 60 + next() * 120
    let dx = 180 + next() * 200
    let dy = 60 + next() * 120
    return Layout(
        clickTarget: CGRect(x: tx, y: ty, width: 90, height: 60),
        dragSource: CGRect(x: sx, y: sy, width: 56, height: 56),
        dragDestination: CGRect(x: dx, y: dy, width: 80, height: 80),
        scrollMaxPx: 900
    )
}

final class SandboxView: NSView {
    var layout = layoutForSeed(1)
    var cursorX: CGFloat = CONTENT_W / 2
    var cursorY: CGFloat = CONTENT_H / 2
    var buttonDown = false
    var scrollOffset: CGFloat = 0
    var clickFlashUntil: CFTimeInterval = 0
    var targetHits = 0
    var dragDone = false
    // The draggable object is actually drawn at (and retained at) its current
    // position; it moves only while an armed drag holds it.
    var dragObjectPos: CGPoint
    override init(frame frameRect: NSRect) {
        dragObjectPos = CGPoint(x: layout.dragSource.midX, y: layout.dragSource.midY)
        super.init(frame: frameRect)
    }
    required init?(coder: NSCoder) { fatalError("unused") }
    func resetObject() {
        dragObjectPos = CGPoint(x: layout.dragSource.midX, y: layout.dragSource.midY)
    }

    override var isFlipped: Bool { return false } // +y up, matching pad space
    override var acceptsFirstResponder: Bool { return false }

    override func draw(_ dirtyRect: NSRect) {
        NSColor(calibratedWhite: 0.94, alpha: 1).setFill()
        bounds.fill()

        // Scroll content region: right-hand clipped strip of colored rows.
        let scrollFrame = CGRect(x: CONTENT_W - 260, y: 0, width: 240, height: CONTENT_H)
        NSGraphicsContext.saveGraphicsState()
        NSBezierPath(rect: scrollFrame).addClip()
        NSColor(calibratedWhite: 0.88, alpha: 1).setFill()
        scrollFrame.fill()
        let rowH: CGFloat = 40
        let firstRow = Int(scrollOffset / rowH)
        for row in firstRow..<(firstRow + Int(CONTENT_H / rowH) + 2) {
            let y = CGFloat(row) * rowH - scrollOffset
            let hue = CGFloat(row % 8) / 8.0
            NSColor(calibratedHue: hue, saturation: 0.5, brightness: 0.85, alpha: 1).setFill()
            NSBezierPath(rect: CGRect(x: scrollFrame.minX + 8, y: y + 4, width: scrollFrame.width - 16, height: rowH - 8)).fill()
            ("row \(row)" as NSString).draw(at: NSPoint(x: scrollFrame.minX + 16, y: y + 14),
                withAttributes: [.font: NSFont.systemFont(ofSize: 13), .foregroundColor: NSColor.black])
        }
        NSGraphicsContext.restoreGraphicsState()
        NSColor.darkGray.setStroke()
        NSBezierPath(rect: scrollFrame).stroke()

        // Click target (green flash after a completed click on it).
        let flashing = CACurrentMediaTime() < clickFlashUntil
        (flashing ? NSColor.systemGreen : NSColor.systemBlue).setFill()
        NSBezierPath(roundedRect: layout.clickTarget, xRadius: 8, yRadius: 8).fill()
        ("click target" as NSString).draw(at: NSPoint(x: layout.clickTarget.minX + 6, y: layout.clickTarget.midY - 8),
            withAttributes: [.font: NSFont.systemFont(ofSize: 12), .foregroundColor: NSColor.white])

        // Drag source outline (original position marker) and the draggable
        // object itself at its live position.
        NSColor.systemOrange.withAlphaComponent(0.35).setStroke()
        let srcOutline = NSBezierPath(ovalIn: layout.dragSource)
        srcOutline.lineWidth = 2
        srcOutline.stroke()
        let objectRect = CGRect(x: dragObjectPos.x - layout.dragSource.width / 2,
                                y: dragObjectPos.y - layout.dragSource.height / 2,
                                width: layout.dragSource.width, height: layout.dragSource.height)
        NSColor.systemOrange.setFill()
        NSBezierPath(ovalIn: objectRect).fill()
        let dash = NSBezierPath(rect: layout.dragDestination)
        dash.setLineDash([6, 4], count: 2, phase: 0)
        (dragDone ? NSColor.systemGreen : NSColor.systemPurple).setStroke()
        dash.lineWidth = 3
        dash.stroke()

        // Virtual cursor: ring when pressed, dot always.
        NSColor.systemRed.setStroke()
        let ring = NSBezierPath(ovalIn: CGRect(x: cursorX - 9, y: cursorY - 9, width: 18, height: 18))
        ring.lineWidth = buttonDown ? 4 : 2
        ring.stroke()
        NSColor.systemRed.setFill()
        NSBezierPath(ovalIn: CGRect(x: cursorX - 3, y: cursorY - 3, width: 6, height: 6)).fill()
    }
}

final class Controller: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var view: SandboxView!
    var processor: TouchProcessor!
    var seed: UInt64 = 1
    var frameId: UInt64 = 0
    var lastSimTimeS: Double = 0
    var releaseCount = 0
    var targetHits = 0
    var dragging = false
    // Position of the current press's down-transition; a click-target hit
    // requires the press to have started inside the target as well as the
    // release to land inside it. Cleared on release/cancel/stop/reset.
    var pressOrigin: CGPoint?
    var lastInputWall: CFTimeInterval?
    var out = FileHandle.standardOutput
    // Role boundary: the privileged evaluator query requires this launch-only
    // flag. A normal actor launch cannot query task truth. This is an
    // enforced protocol role, not an OS security claim.
    let evaluatorEnabled = CommandLine.arguments.contains("--evaluator")
    // Diagnostics-only mode: enabled solely via launch flags, never by an
    // actor command. Probes write only a fixed canary name under a fixed
    // caller-designated fresh root and connect only to validated loopback.
    let diagnosticsEnabled = CommandLine.arguments.contains("--diagnostics")
    let canaryRoot: String? = {
        guard let i = CommandLine.arguments.firstIndex(of: "--canary-root"),
              i + 1 < CommandLine.arguments.count else { return nil }
        return CommandLine.arguments[i + 1]
    }()
    let canaryName = "sandbox-canary.txt"

    func applicationDidFinishLaunching(_ notification: Notification) {
        processor = TouchProcessor(config: PadConfig(contentWidthPx: Double(CONTENT_W), contentHeightPx: Double(CONTENT_H)))
        view = SandboxView(frame: NSRect(x: 0, y: 0, width: CONTENT_W, height: CONTENT_H))
        window = NSWindow(contentRect: view.frame, styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "TrackpadSandbox (bounded native test GUI)"
        window.contentView = view
        window.center()
        window.makeKeyAndOrderFront(nil)
        emit(["type": "hello", "app": "TrackpadSandbox", "version": "0.3.0",
              "diagnostics": diagnosticsEnabled, "evaluator": evaluatorEnabled,
              "content_size": ["w": Double(CONTENT_W), "h": Double(CONTENT_H)]])
        // Host monotonic watchdog: a stalled transport (no touch frames for
        // >0.8 s wall time) cancels any held button/drag without a click.
        Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self = self, let last = self.lastInputWall else { return }
            let idle = CACurrentMediaTime() - last
            if self.processor.wallClockIdleCancel(idleS: idle) {
                self.dragging = false
                self.pressOrigin = nil
                self.syncView()
            }
        }
        DispatchQueue.global().async {
            while let line = readLine(strippingNewline: true) {
                DispatchQueue.main.sync { self.handleLine(line) }
            }
            DispatchQueue.main.sync { self.finish(status: 0, reason: "eof") }
        }
    }

    func emit(_ obj: [String: Any]) {
        if let data = try? JSONSerialization.data(withJSONObject: obj), let s = String(data: data, encoding: .utf8) {
            out.write((s + "\n").data(using: .utf8)!)
        }
    }

    func emitError(_ code: String, _ detail: String) {
        emit(["type": "error", "code": code, "detail": detail])
    }

    func finish(status: Int32, reason: String) {
        processor.releaseAll()
        syncView()
        emit(["type": "bye", "reason": reason])
        NSApp.terminate(nil)
    }

    func syncView() {
        view.cursorX = CGFloat(processor.cursorXPx)
        view.cursorY = CGFloat(processor.cursorYPx)
        view.buttonDown = processor.buttonDown
        view.scrollOffset = CGFloat(processor.scrollOffsetPx)
        view.needsDisplay = true
    }

    func handleLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let any = try? JSONSerialization.jsonObject(with: data),
              let msg = any as? [String: Any],
              let cmd = msg["cmd"] as? String else {
            emitError("malformed", "line is not a JSON object with string cmd")
            return
        }
        let allowed: Set<String> = ["reset", "touch", "capture", "state", "stop", "probe_file", "probe_net"]
        guard allowed.contains(cmd) else { emitError("unknown_cmd", cmd); return }

        switch cmd {
        case "reset":
            guard let sv = msg["seed"], !TouchProcessor.isBoolValue(sv), let seedVal = sv as? Int, seedVal >= 0 else {
                emitError("bad_seed", "seed must be a non-negative int"); return
            }
            seed = UInt64(seedVal)
            view.layout = layoutForSeed(seed)
            view.targetHits = 0; view.dragDone = false
            view.resetObject()
            dragging = false; releaseCount = 0; targetHits = 0
            pressOrigin = nil
            lastInputWall = nil
            processor.resetCursor()
            syncView()
            emit(["type": "reset_ok", "seed": seedVal])
        case "touch":
            for key in msg.keys where !["cmd", "schema_version", "sequence", "simulation_time_s", "contacts"].contains(key) {
                emitError("unknown_field", key); return
            }
            do {
                let events = try processor.handleTouch(
                    schemaVersion: msg["schema_version"], sequence: msg["sequence"],
                    timeS: msg["simulation_time_s"], rawContacts: msg["contacts"])
                lastInputWall = CACurrentMediaTime()
                applyEvents(events)
                if let t = try? TouchProcessor.strictDouble(msg["simulation_time_s"], "simulation_time_s") {
                    lastSimTimeS = t
                }
                syncView()
                // Acknowledgement carries acceptance/sequence only: no cursor,
                // layout, or task-success truth (that stays in privileged state).
                emit(["type": "touch_ok", "sequence": msg["sequence"] as? Int ?? 0])
            } catch let e as TouchRejection {
                emitError("touch_rejected", describe(e))
            } catch {
                emitError("touch_rejected", "unknown")
            }
        case "capture":
            capture()
        case "state":
            // Privileged evaluator query, gated by the launch-only
            // --evaluator role flag; a normal actor launch is rejected. Task
            // state is deliberately separate from actor pixels.
            guard evaluatorEnabled else {
                emitError("evaluator_required", "state requires --evaluator at launch"); return
            }
            emit(["type": "state", "privileged": true,
                  "cursor": ["x": processor.cursorXPx, "y": processor.cursorYPx],
                  "button_down": processor.buttonDown,
                  "press_suppressed": processor.pressSuppressed,
                  "scroll_offset_px": processor.scrollOffsetPx,
                  "release_count": releaseCount, "target_hits": targetHits,
                  "drag_done": view.dragDone,
                  "drag_object": ["x": Double(view.dragObjectPos.x), "y": Double(view.dragObjectPos.y)],
                  "seed": Int(seed), "last_simulation_time_s": lastSimTimeS,
                  "click_target": rectDict(view.layout.clickTarget),
                  "drag_source": rectDict(view.layout.dragSource),
                  "drag_destination": rectDict(view.layout.dragDestination)])
        case "stop":
            processor.releaseAll() // full gesture-state reset; never completes a task
            dragging = false
            pressOrigin = nil
            lastInputWall = nil
            syncView()
            emit(["type": "stopped"])
        case "probe_file":
            guard diagnosticsEnabled, let root = canaryRoot else {
                emitError("diagnostics_disabled", "probe_file requires --diagnostics --canary-root at launch"); return
            }
            let path = (root as NSString).appendingPathComponent(canaryName)
            do {
                try "sandbox-escape-probe".write(toFile: path, atomically: false, encoding: .utf8)
                emit(["type": "probe_file", "path": path, "allowed": true])
            } catch let e as NSError {
                emit(["type": "probe_file", "path": path, "allowed": false,
                      "error_domain": e.domain, "error_code": e.code])
            }
        case "probe_net":
            guard diagnosticsEnabled else {
                emitError("diagnostics_disabled", "probe_net requires --diagnostics at launch"); return
            }
            guard let pv = msg["port"], !TouchProcessor.isBoolValue(pv), let port = pv as? Int, (1...65535).contains(port) else {
                emitError("bad_probe", "port must be an int in 1...65535"); return
            }
            emit(probeNet(port: UInt16(port)))
        default:
            break
        }
    }

    func rectDict(_ r: CGRect) -> [String: Double] {
        ["x": Double(r.minX), "y": Double(r.minY), "w": Double(r.width), "h": Double(r.height)]
    }

    // Task bookkeeping from gesture events. A drag arms ONLY on a press
    // down-transition whose cursor is inside the draggable object; pressing
    // elsewhere and sliding through the object never arms it. Completion is
    // decided by the object's observed placement at release, and stop/expiry/
    // watchdog cancels never complete anything.
    func applyEvents(_ events: GestureEvents) {
        if events.pressBegan {
            let p = CGPoint(x: processor.cursorXPx, y: processor.cursorYPx)
            pressOrigin = p
            let objRect = CGRect(x: view.dragObjectPos.x - view.layout.dragSource.width / 2,
                                 y: view.dragObjectPos.y - view.layout.dragSource.height / 2,
                                 width: view.layout.dragSource.width,
                                 height: view.layout.dragSource.height)
            dragging = objRect.contains(p)
        }
        if dragging && processor.buttonDown {
            view.dragObjectPos = CGPoint(x: processor.cursorXPx, y: processor.cursorYPx)
        }
        if let rel = events.releaseAt {
            releaseCount += 1
            let p = CGPoint(x: rel.x, y: rel.y)
            // A target click requires the press to have started inside the
            // target AND the release to land inside it; a down-outside then
            // slide-in is not a target activation.
            if view.layout.clickTarget.contains(p),
               let origin = pressOrigin, view.layout.clickTarget.contains(origin) {
                targetHits += 1
                view.targetHits = targetHits
                view.clickFlashUntil = CACurrentMediaTime() + 0.8
            }
            pressOrigin = nil
            if dragging {
                if view.layout.dragDestination.contains(view.dragObjectPos) {
                    view.dragDone = true
                }
                dragging = false
            }
        }
        if !processor.buttonDown {
            dragging = false
            if events.releaseAt == nil {
                pressOrigin = nil // silent cancel (mode change/expiry/watchdog)
            }
        }
    }

    func capture() {
        frameId += 1
        view.displayIfNeeded()
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
            emitError("capture_failed", "no bitmap rep"); return
        }
        view.cacheDisplay(in: view.bounds, to: rep)
        guard let png = rep.representation(using: .png, properties: [:]) else {
            emitError("capture_failed", "png encoding failed"); return
        }
        emit(["type": "observation", "frame_id": frameId,
              "width": rep.pixelsWide, "height": rep.pixelsHigh,
              "monotonic_time_s": CACurrentMediaTime(),
              "simulation_time_s": lastSimTimeS,
              "png_base64": png.base64EncodedString()])
    }

    // Bounded sandbox probes: attempt exactly one designated temp file outside
    // the container, and one localhost TCP connect. Results are reported as
    // measured restrictions; entitlements are reported separately by the client.
    func probeNet(port: UInt16) -> [String: Any] {
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        if fd < 0 { return ["type": "probe_net", "allowed": false, "stage": "socket", "errno": errno] }
        let flags = fcntl(fd, F_GETFL)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        var a = addr
        let rc = withUnsafePointer(to: &a) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        if rc == 0 { close(fd); return ["type": "probe_net", "allowed": true] }
        if errno == EINPROGRESS {
            usleep(200000)
            var soerr: Int32 = 0
            var len = socklen_t(MemoryLayout<Int32>.size)
            getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len)
            close(fd)
            if soerr == 0 { return ["type": "probe_net", "allowed": true] }
            if soerr == EINPROGRESS { return ["type": "probe_net", "allowed": false, "errno": soerr, "stage": "timeout"] }
            return ["type": "probe_net", "allowed": false, "errno": soerr]
        }
        let e = errno
        close(fd)
        return ["type": "probe_net", "allowed": false, "errno": e]
    }

    func describe(_ e: TouchRejection) -> String {
        switch e {
        case .unknownField(let f): return "unknown field: \(f)"
        case .wrongSchema(let d): return d
        case .badType(let d): return d
        case .nonFinite(let f): return "non-finite/invalid: \(f)"
        case .outOfOrderSequence(let exp, let got): return "sequence expected \(exp) got \(got)"
        case .nonMonotonicTime(let last, let got): return "time \(got) < last \(last)"
        case .tooManyContacts(let n): return "too many contacts: \(n)"
        case .duplicateFingerId(let f): return "duplicate finger_id: \(f)"
        case .emptyFingerId: return "empty finger_id"
        case .contactOutOfRange(let f): return "contact out of pad range: \(f)"
        case .excessiveDelta(let f): return "excessive delta: \(f)"
        }
    }

    func applicationWillTerminate(_ notification: Notification) {}
}

let app = NSApplication.shared
let controller = Controller()
app.delegate = controller
app.setActivationPolicy(.regular)
app.run()
