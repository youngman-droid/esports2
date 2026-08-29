#!/bin/sh
# Full refresh: markets (discover/backfill), Oracle's Elixir CSV + draft tables,
# gol.gg incremental scrape, then alignment + odds-free models + live model.
# Usage: sh scripts/update.sh [--no-record]   (logs: data/update.log)
set -u
cd "$(dirname "$0")/.."
OE_2026_ID="1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"
log(){ echo "$(date -u +%FT%TZ) update: $*"; }

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
wait $PA; log "phase A done (rc=$?)"
wait $PB; log "phase B done (rc=$?)"
wait $PC; log "phase C done (rc=$?)"
python3 scripts/feed_backfill.py --workers 6 > data/update_feed.log 2>&1; log "feed hp backfill rc=$?"
log "phase D: align -> wpa -> draftfree -> wpx prep/build -> live model"
python3 -m lol_ticker align > data/update_align.log 2>&1; log "align rc=$?"
python3 -m lol_ticker wpa > data/update_wpa.log 2>&1; log "wpa rc=$?"
python3 -m lol_ticker draftfree > data/update_draftfree.log 2>&1; log "draftfree rc=$?"
( python3 -m lol_ticker wpx prep && python3 -m lol_ticker wpx build && python3 -c "from lol_ticker import wpx; wpx.fit_full()" ) > data/update_wpx.log 2>&1; log "wpx/live model rc=$?"
python3 scripts/wpx_eval.py > data/update_wpx_eval.log 2>&1; log "wpx eval/results.json rc=$?"
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
