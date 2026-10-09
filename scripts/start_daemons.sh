#!/bin/sh
# Start the long-running daemons that are not already running, each in its
# own session (setsid) so they outlive the shell, editor or agent session
# that launched them.  A plain `nohup ... &` stays in the caller's process
# group and dies when that group is torn down -- both daemons were found dead
# that way on 2026-09-27 after a manual refresh on 09-15, with nothing logged.
#
# Usage: sh scripts/start_daemons.sh [record|shadow|dashboard|all]   (default: all)
# Logs: data/record.log, data/shadow.log, data/dashboard.log (appended).
#
# Normally launchd owns the daemons (sh scripts/install_agents.sh): a daemon
# whose com.lolticker.<name> agent is loaded is left to launchd (KeepAlive).
# This script covers machines without the agents and manual restarts.
# The repo's .venv is used when present.
set -eu
cd "$(dirname "$0")/.."
PATH="$(pwd)/.venv/bin:$PATH"; export PATH
what="${1:-all}"

start(){
  name="$1"; pattern="$2"; logfile="$3"; label="com.lolticker.$4"; shift 4
  if launchctl print "gui/$(id -u)/$label" > /dev/null 2>&1; then
    echo "$name managed by launchd ($label)"
    return 0
  fi
  if pgrep -f "$pattern" > /dev/null; then
    echo "$name already running (pid $(pgrep -f "$pattern" | head -1))"
    return 0
  fi
  pid=$(python3 - "$logfile" "$@" <<'EOF'
import os, subprocess, sys
logfile, argv = sys.argv[1], sys.argv[2:]
with open(logfile, "ab") as out:
    p = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=os.getcwd(),
                         start_new_session=True)
print(p.pid)
EOF
)
  echo "$name started (pid $pid)"
}

case "$what" in
  record|all)
    start "record daemon" "lol_ticker record" data/record.log record \
      python3 -m lol_ticker record ;;
esac
case "$what" in
  shadow|all)
    start "prospective shadow recorder" "lol_ticker shadow record" data/shadow.log shadow \
      python3 -m lol_ticker shadow record ;;
esac
case "$what" in
  dashboard)
    start "dashboard" "lol_ticker dashboard" data/dashboard.log dashboard \
      python3 -m lol_ticker dashboard ;;
esac
case "$what" in
  record|shadow|dashboard|all) ;;
  *) echo "usage: $0 [record|shadow|dashboard|all]" >&2; exit 2 ;;
esac
