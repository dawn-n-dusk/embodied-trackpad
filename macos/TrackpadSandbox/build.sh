#!/bin/bash
# Builds TrackpadSandbox.app into .local/build/ (ignored) and ad-hoc signs it
# with ONLY com.apple.security.app-sandbox. No network, no dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/.local/build/TrackpadSandbox.app"
mkdir -p "$OUT/Contents/MacOS"
swiftc -O -module-name TrackpadSandbox \
  "$ROOT/macos/TrackpadSandbox/Sources/main.swift" \
  "$ROOT/macos/TrackpadSandbox/Sources/DeviceModel.swift" \
  -o "$OUT/Contents/MacOS/TrackpadSandbox"
cp "$ROOT/macos/TrackpadSandbox/Info.plist" "$OUT/Contents/Info.plist"
codesign --force --sign - \
  --entitlements "$ROOT/macos/TrackpadSandbox/TrackpadSandbox.entitlements" \
  "$OUT"
echo "built: $OUT"
codesign -dv --entitlements :- "$OUT" 2>&1 | grep -A2 'com.apple.security' || true
