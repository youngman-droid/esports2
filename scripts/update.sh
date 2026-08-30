#!/bin/sh
# Full refresh: markets (discover/backfill), Oracle's Elixir CSV + draft tables,
# gol.gg incremental scrape, then alignment + odds-free models + live model.
# Usage: sh scripts/update.sh [--no-record]   (logs: data/update.log)
set -eu
cd "$(dirname "$0")/.."
OE_2026_ID="1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"
log(){ echo "$(date -u +%FT%TZ) update: $*"; }
wait_phase(){
  phase="$1"
  pid="$2"
  if wait "$pid"; then
    log "phase ${phase} done"
  else
    rc=$?
    log "phase ${phase} FAILED (rc=${rc}); stopping before model refresh"
    exit "$rc"
  fi
}

log "phase A: markets (discover -> backfill)"
( python3 -m lol_ticker discover && python3 -m lol_ticker backfill ) > data/update_markets.log 2>&1 &
PA=$!
log "phase B: Oracle's Elixir (download 2026 csv -> draftload)"
( python3 scripts/drive_download.py "${OE_2026_ID}" data/oe/oe_2026.csv.tmp \
    && mv data/oe/oe_2026.csv.tmp data/oe/oe_2026.csv \
    && python3 -m lol_ticker draftload ) > data/update_oe.log 2>&1 &
PB=$!
log "phase C: gol.gg incremental scrape"
( python3 -m lol_ticker golgg --seasons S16 --regions major --since "$(date -u -v-21d +%F 2>/dev/null || date -u -d '21 days ago' +%F)" --workers 8 ) > data/update_golgg.log 2>&1 &
PC=$!
wait_phase A "$PA"
wait_phase B "$PB"
wait_phase C "$PC"
python3 scripts/feed_backfill.py --workers 6 > data/update_feed.log 2>&1
log "feed hp backfill done"
log "phase D: align -> wpa -> draftfree -> wpx prep/build -> live model"
python3 -m lol_ticker align > data/update_align.log 2>&1
python3 -m lol_ticker wpa > data/update_wpa.log 2>&1
python3 -m lol_ticker draftfree > data/update_draftfree.log 2>&1
( python3 -m lol_ticker wpx prep && python3 -m lol_ticker wpx build \
    && python3 -m lol_ticker wpx fit ) > data/update_wpx.log 2>&1
log "staged live stack passed validation and was promoted"
python3 scripts/wpx_eval.py > data/update_wpx_eval.log 2>&1
log "wpx evaluation complete"
if [ "${1:-}" != "--no-record" ]; then
  if ! pgrep -f "lol_ticker record" > /dev/null; then
    nohup python3 -m lol_ticker record > data/record.log 2>&1 &
    log "record daemon started (pid $!)"
  else
    log "record daemon already running"
  fi
  if ! pgrep -f "lol_ticker shadow record" > /dev/null; then
    nohup python3 -m lol_ticker shadow record > data/shadow.log 2>&1 &
    log "prospective shadow recorder started (pid $!)"
  else
    log "prospective shadow recorder already running"
  fi
fi
log "ALL DONE"
