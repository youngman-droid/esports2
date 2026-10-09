#!/bin/sh
# Rotate the long-running daemons' logs (run nightly by update.sh).
# A log over $MAX_MB is gzipped to data/logs/<name>.<UTC stamp>.log.gz and
# truncated in place: the daemons hold it open in append mode (launchd's
# StandardOutPath, start_daemons.sh's "ab"), so they keep writing to the
# same, now empty, file.  The newest $KEEP archives per log are kept.
set -eu
cd "$(dirname "$0")/.."
MAX_MB="${MAX_MB:-20}"
KEEP="${KEEP:-6}"
mkdir -p data/logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
for name in record shadow dashboard watchdog; do
  f="data/$name.log"
  [ -f "$f" ] || continue
  [ "$(stat -f %z "$f")" -gt $((MAX_MB * 1024 * 1024)) ] || continue
  gzip -c "$f" > "data/logs/$name.$stamp.log.gz"
  : > "$f"
  echo "rotated $f -> data/logs/$name.$stamp.log.gz"
  ls -1t data/logs/"$name".*.log.gz | tail -n +$((KEEP + 1)) | while read -r old; do rm -f "$old"; done
done
