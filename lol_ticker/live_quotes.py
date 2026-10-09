"""Bounded, nonblocking quote collection for immutable shadow forecasts.

Workers never touch the recorder's database connection. A cache miss is a
missing benchmark; rows are never retroactively paired with later quotes.
"""
from copy import deepcopy
import logging
import threading
import time

from . import live

log = logging.getLogger("live_quotes")
MAX_AGE_S = 15.0
LOOKUP_BUDGET_S = 12.0


class QuoteCache:
    def __init__(self, fetch=None, max_workers=4, max_age_s=MAX_AGE_S,
                 budget_s=LOOKUP_BUDGET_S, clock=time.time):
        self.fetch = fetch or live.market_prices
        self.clock = clock
        self.max_age_s = max_age_s
        self.budget_s = budget_s
        self.slots = threading.BoundedSemaphore(max_workers)
        self.lock = threading.Lock()
        self.entries = {}
        self.running = set()
        self.anchors = {}

    @staticmethod
    def key(game, start_ts):
        # Exact attempt anchor prevents quotes crossing a remake or side swap.
        return (str(game["game_id"]), int(start_ts), tuple(game["teams"]),
                game.get("number") or 1, bool(game.get("deciding", False)))

    def request(self, game, start_ts):
        key = self.key(game, start_ts)
        with self.lock:
            self.anchors[str(game["game_id"])] = int(start_ts)
            recent = self.entries.get(key)
            if recent and self.clock() - recent[0] < 5:
                return False
            if key in self.running or not self.slots.acquire(blocking=False):
                return False
            self.running.add(key)
            # Bound memory while preserving active lookups.
            cutoff = self.clock() - 3600
            self.entries = {k: v for k, v in self.entries.items() if v[0] >= cutoff}
        thread = threading.Thread(target=self._refresh, args=(key,), daemon=True)
        thread.start()
        return True

    def prefetch(self, game):
        with self.lock:
            start = self.anchors.get(str(game["game_id"]))
        if start is not None:
            self.request(game, start)

    def _refresh(self, key):
        _, start, teams, number, deciding = key
        try:
            with live.request_deadline(self.budget_s):
                quotes = self.fetch(None, list(teams), number, deciding=deciding,
                                    game_start_ts=start)
            with self.lock:
                self.entries[key] = (self.clock(), deepcopy(quotes))
        except Exception as exc:
            log.warning("quote cache refresh failed: %s", exc)
        finally:
            with self.lock:
                self.running.discard(key)
            self.slots.release()

    def snapshot(self, game, start_ts):
        key = self.key(game, start_ts)
        now = self.clock()
        with self.lock:
            _, quotes = self.entries.get(key, (None, {}))
            quotes = deepcopy(quotes)
        out = {}
        for platform, row in quotes.items():
            stamp = row.get("captured_ts_ms")
            age = now - float(stamp) / 1000 if stamp is not None else None
            if age is not None and 0 <= age <= self.max_age_s:
                out[platform] = dict(row, quote_age_s=age, collection="background_cache")
        return out


CACHE = QuoteCache()
