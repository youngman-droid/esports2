#!/bin/sh
# Install (or remove) the launchd agents that keep the ticker running across
# reboots, generated for wherever this checkout lives:
#   com.lolticker.record     book/trade recorder           KeepAlive
#   com.lolticker.shadow     prospective shadow recorder   KeepAlive
#   com.lolticker.dashboard  web dashboard :8090           KeepAlive
#   com.lolticker.watchdog   health alert pass             every 5 min
#   com.lolticker.update     nightly data refresh          06:30 local
#
# Usage: sh scripts/install_agents.sh [install|uninstall]   (default: install)
# Needs the repo's venv (.venv, see README).  Set LOL_TICKER_NTFY_URL before
# installing to have the watchdog also push alerts to that URL.
#
# macOS privacy protection (TCC) denies launchd-spawned processes access to
# ~/Documents, ~/Desktop and ~/Downloads, so the checkout must live elsewhere
# (e.g. ~/Developer/esports2); the script refuses to install from those.
set -eu
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
UID_N="$(id -u)"
AGENTS="$HOME/Library/LaunchAgents"
LABELS="record shadow dashboard watchdog update"

unload(){
  for name in $LABELS; do
    launchctl bootout "gui/$UID_N/com.lolticker.$name" 2>/dev/null || true
    rm -f "$AGENTS/com.lolticker.$name.plist"
  done
}

if [ "${1:-install}" = "uninstall" ]; then
  unload; echo "removed com.lolticker.* agents"; exit 0
fi

case "$ROOT" in
  "$HOME/Documents"*|"$HOME/Desktop"*|"$HOME/Downloads"*)
    echo "refusing: $ROOT is privacy-protected; launchd jobs cannot read it." >&2
    echo "move the checkout (e.g. to ~/Developer/esports2) and rerun." >&2
    exit 1 ;;
esac
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "missing $PY -- create the venv first (README: Setup)" >&2; exit 1; }

xml(){ printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

# plist <name> <log> <schedule-xml> <program args...>
plist(){
  name="$1"; log="$2"; schedule="$3"; shift 3
  args=""
  for a in "$@"; do args="$args        <string>$(xml "$a")</string>
"; done
  ntfy=""
  if [ -n "${LOL_TICKER_NTFY_URL:-}" ]; then
    ntfy="        <key>LOL_TICKER_NTFY_URL</key><string>$(xml "$LOL_TICKER_NTFY_URL")</string>
"
  fi
  cat > "$AGENTS/com.lolticker.$name.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.lolticker.$name</string>
    <key>ProgramArguments</key>
    <array>
$args    </array>
    <key>WorkingDirectory</key><string>$(xml "$ROOT")</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>$(xml "$ROOT/.venv/bin"):/opt/homebrew/bin:/opt/homebrew/opt/postgresql@18/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
${ntfy}    </dict>
$schedule
    <key>StandardOutPath</key><string>$(xml "$ROOT/data/$log")</string>
    <key>StandardErrorPath</key><string>$(xml "$ROOT/data/$log")</string>
</dict>
</plist>
EOF
}

KEEP='    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>30</integer>'
EVERY5='    <key>RunAtLoad</key><true/>
    <key>StartInterval</key><integer>300</integer>'
NIGHTLY='    <key>StartCalendarInterval</key><dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer></dict>'

mkdir -p "$AGENTS" "$ROOT/data"
unload
# Stop daemons started by hand (start_daemons.sh) so launchd's copies do not
# run beside them.
for pattern in "lol_ticker record" "lol_ticker shadow record" "lol_ticker dashboard"; do
  pkill -f "$pattern" 2>/dev/null || true
done
sleep 2
plist record record.log "$KEEP" "$PY" -m lol_ticker record
plist shadow shadow.log "$KEEP" "$PY" -m lol_ticker shadow record
plist dashboard dashboard.log "$KEEP" "$PY" -m lol_ticker dashboard
plist watchdog watchdog.log "$EVERY5" "$PY" -m lol_ticker watchdog
plist update update.log "$NIGHTLY" /bin/sh "$ROOT/scripts/update.sh"
for name in $LABELS; do
  plutil -lint -s "$AGENTS/com.lolticker.$name.plist"
  launchctl bootstrap "gui/$UID_N" "$AGENTS/com.lolticker.$name.plist"
done
echo "installed com.lolticker.{$(echo $LABELS | tr ' ' ',')} for $ROOT"
