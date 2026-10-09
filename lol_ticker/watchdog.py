"""Recorder watchdog: one health pass, meant to run every few minutes (launchd).

Problems it reports:
  * a long-running daemon (``record`` / ``shadow record``) is not running;
  * a LoL Esports game is in progress but no order book has been stored for
    ``LIVE_STALE_S`` — live L2 depth cannot be backfilled later;
  * no order book stored for ``IDLE_STALE_S`` even with nothing live.

Alerts go to a macOS notification and, when ``$LOL_TICKER_NTFY_URL`` is set
(e.g. https://ntfy.sh/<private-topic>), to that URL as a push.  A problem is
re-announced at most every ``REPEAT_S``; a "recovered" note follows when it
clears.  State lives in data/watchdog_state.json.
"""
import json
import logging
import os
import subprocess
import time
import urllib.request

from . import config

log = logging.getLogger("watchdog")

LIVE_STALE_S = 15 * 60
IDLE_STALE_S = 6 * 3600
REPEAT_S = 3600
STATE_PATH = os.path.join(config.REPO_ROOT, "data", "watchdog_state.json")
DAEMONS = {"record": "lol_ticker record", "shadow": "lol_ticker shadow record"}


def find_problems(running, book_age_s, live_games):
    """Pure check.  ``running``: daemon name -> bool (None = unknown);
    ``book_age_s``: seconds since the newest stored book (None = none);
    ``live_games``: number of in-progress games (None = schedule unreachable)."""
    problems = {}
    for name, alive in running.items():
        if alive is False:
            problems["daemon:" + name] = "%s daemon is not running" % name
    if book_age_s is None:
        problems["books"] = "no order book has ever been stored"
    elif live_games and book_age_s > LIVE_STALE_S:
        problems["books"] = "%d game(s) live but no book stored for %d min" % (
            live_games, book_age_s // 60)
    elif book_age_s > IDLE_STALE_S:
        problems["books"] = "no book stored for %.1f h" % (book_age_s / 3600)
    return problems


def plan_alerts(problems, state, now):
    """Which messages to send and the next state.  Returns (messages, state)."""
    messages, new_state = [], {}
    for key, text in problems.items():
        prev = state.get(key)
        if prev is None or now - prev["last_alert"] >= REPEAT_S:
            messages.append(text)
            new_state[key] = {"since": prev["since"] if prev else now, "last_alert": now}
        else:
            new_state[key] = prev
    for key, prev in state.items():
        if key not in problems:
            messages.append("recovered: %s (after %d min)" % (key, (now - prev["since"]) // 60))
    return messages, new_state


def _running():
    out = {}
    for name, pattern in DAEMONS.items():
        try:
            out[name] = subprocess.run(["pgrep", "-f", pattern], capture_output=True,
                                       timeout=5).returncode == 0
        except (OSError, subprocess.SubprocessError):
            out[name] = None
    return out


def _book_age(conn):
    row = conn.execute(
        "SELECT extract(epoch from now() - max(ts)) AS age FROM book_snapshots").fetchone()
    conn.rollback()
    return int(row["age"]) if row and row["age"] is not None else None


def _live_count():
    from . import live
    try:
        return len(live.live_games(probe=False))
    except Exception as e:   # schedule outage must not hide a dead recorder
        log.warning("live schedule unavailable: %s", e)
        return None


def notify(text, title="LoL ticker"):
    try:
        subprocess.run(["osascript", "-e", "display notification %s with title %s" % (
            json.dumps(text), json.dumps(title))], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("macOS notification failed: %s", e)
    url = os.environ.get("LOL_TICKER_NTFY_URL")
    if url:
        try:
            req = urllib.request.Request(url, data=text.encode(), headers={"Title": title})
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as e:
            log.warning("push notification failed: %s", e)


def _load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def run_once(conn, now=None):
    now = int(now if now is not None else time.time())
    problems = find_problems(_running(), _book_age(conn), _live_count())
    messages, state = plan_alerts(problems, _load_state(), now)
    for text in messages:
        log.warning("%s", text)
        notify(text)
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)
    if not problems:
        log.info("ok")
    return problems
