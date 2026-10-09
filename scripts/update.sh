#!/bin/sh
# Data refresh: markets, Oracle's Elixir, gol.gg, alignment and source priors.
# Live GAM refresh is gated while the frozen corrected-input candidate is scored.
# Usage: sh scripts/update.sh [--no-record]   (logs: data/update.log)
# Scheduled nightly by the com.lolticker.update agent (scripts/install_agents.sh, 06:30 local).
set -eu
cd "$(dirname "$0")/.."
PATH="$(pwd)/.venv/bin:$PATH"; export PATH   # the repo's venv python3 when present
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
# Oracle's Elixir is a secondary source: its Drive download hits quota for
# weeks at a time and the downloader then refuses the <1 MB stub, leaving the
# previous CSV in place.  That must not block the gol.gg scrape, the recency
# Elo, or shadow outcome resolution, so phase B only warns.
wait_phase_soft(){
  phase="$1"
  pid="$2"
  if wait "$pid"; then
    log "phase ${phase} done"
  else
    rc=$?
    log "phase ${phase} FAILED (rc=${rc}); continuing with the previous data (non-fatal)"
  fi
}

sh scripts/rotate_logs.sh | while read -r line; do log "$line"; done
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
# Solo-queue pair prior: pages are re-fetched at most weekly while a patch is
# live (~1,240 requests, ~25 min at 1 req/s), so most nights this is a no-op.
# Runs beside the other phases; a failure only warns (the previous tables stay).
log "phase S: solo-queue pair tables (Lolalytics refresh)"
( python3 -m lol_ticker sqpairs refresh ) > data/update_sqpairs.log 2>&1 &
PS=$!
wait_phase A "$PA"
wait_phase_soft B "$PB"
wait_phase C "$PC"
wait_phase_soft S "$PS"
# Standalone player rankings use the completed local Oracle CSV snapshot.
# They have no connection to the deployed live model or exposure ledger.
log "phase R: lane-informed player rankings"
( OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m lol_ticker.player_ratings build --outcome wins --output data/player_ratings/ratings.json ) > data/update_player_ratings.log 2>&1 &
PR=$!
python3 scripts/feed_backfill.py --workers 6 > data/update_feed.log 2>&1
log "feed hp backfill done"
log "phase D: align -> wpa -> draftfree -> wpx prep"
# Each step logs its own exit code so a failure names the step (set -e then
# stops the run before the model refresh).
step(){
  name="$1"; logfile="$2"; shift 2
  if "$@" > "$logfile" 2>&1; then
    log "${name} rc=0"
  else
    rc=$?
    log "${name} FAILED (rc=${rc}); see ${logfile}"
    exit "$rc"
  fi
}
step align data/update_align.log python3 -m lol_ticker align
step wpa data/update_wpa.log python3 -m lol_ticker wpa
step draftfree data/update_draftfree.log python3 -m lol_ticker draftfree
step "wpx prep" data/update_wpx_prep.log python3 -m lol_ticker wpx prep
# Corrected builds are immutable, explicitly dated artifacts. Default legacy
# fitting/evaluation still reads states.npz, so chaining it after a v2 build
# would silently score different inputs. Model selection, candidate freezing
# and any promotion belong to the explicit corrected-input experiment.
log "live GAM build/fit and historical evaluation gated pending corrected-input candidate evidence"
# Attach outcomes (fresh gol.gg/OE rows, else exchange settlements) to every
# pending shadow forecast and refresh the prospective score.
if python3 -m lol_ticker shadow score >> data/update_shadow.log 2>&1; then
  log "shadow outcomes resolved and scored"
else
  log "shadow resolve/score FAILED (non-fatal; see data/update_shadow.log)"
fi
wait_phase_soft R "$PR"
if [ "${1:-}" != "--no-record" ]; then
  # Detached (own session) so the daemons survive the shell that ran this
  # refresh; see scripts/start_daemons.sh.
  sh scripts/start_daemons.sh all | while read -r line; do log "$line"; done
fi
log "ALL DONE"
