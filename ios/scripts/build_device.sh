#!/usr/bin/env bash
# Builds, signs, and installs the app on a connected device. Automatic signing
# picks the team's cached wildcard development profile, so no Xcode account
# login is needed. The team is read from that profile, so it never appears in
# the repository or on the command line.
#   ios/scripts/build_device.sh <device-name>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
device="${1:?device name, as shown by xcrun devicectl list devices}"
profile_name="${PROFILE_NAME:-iOS Team Provisioning Profile: *}"
"$here/scripts/fetch_llama.sh" >/dev/null
"$here/scripts/build_web.sh" >/dev/null
(cd "$here" && xcodegen generate >/dev/null)

team=""
for f in "$HOME/Library/Developer/Xcode/UserData/Provisioning Profiles"/* "$HOME/Library/MobileDevice/Provisioning Profiles"/*; do
  [[ -f "$f" ]] || continue
  plist="$(security cms -D -i "$f" 2>/dev/null)" || continue
  name="$(/usr/libexec/PlistBuddy -c 'Print :Name' /dev/stdin <<<"$plist" 2>/dev/null || true)"
  if [[ "$name" == "$profile_name" ]]; then
    team="$(/usr/libexec/PlistBuddy -c 'Print :TeamIdentifier:0' /dev/stdin <<<"$plist")"
    break
  fi
done
[[ -n "$team" ]] || { echo "profile '$profile_name' not found" >&2; exit 1; }

xcodebuild -project "$here/DecisionPacman.xcodeproj" -scheme DecisionPacman -configuration Release \
  -destination "platform=iOS,name=$device" -derivedDataPath "$here/build" \
  CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM="$team" \
  build -quiet
app="$here/build/Build/Products/Release-iphoneos/DecisionPacman.app"
udid="$(xcrun devicectl list devices 2>/dev/null | awk -v d="$device" '$1 == d {for (i = 1; i <= NF; i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{4}-/) print $i}' | head -1)"
xcrun devicectl device install app --device "$udid" "$app" >/dev/null
echo "installed on $device ($udid)"
