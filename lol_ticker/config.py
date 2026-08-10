import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PostgreSQL + TimescaleDB connection
PG_DSN = os.environ.get("LOL_TICKER_DSN", "postgresql://localhost:5432/league")

# legacy sqlite file (only used by scripts/migrate_sqlite.py)
SQLITE_PATH = os.environ.get("LOL_TICKER_DB", os.path.join(REPO_ROOT, "data", "league.db"))

# --- API endpoints ---
PM_GAMMA = "https://gamma-api.polymarket.com"
PM_CLOB = "https://clob.polymarket.com"
PM_DATA = "https://data-api.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# --- Discovery scope ---
PM_TAG_SLUG = "league-of-legends"
KALSHI_SERIES = [
    "KXLOLGAME",       # match winner (one market per team)
    "KXLOLMAP",        # map winner
    "KXLOLTOTALMAPS",  # total maps
    "KXLOLTOTAL",      # total maps played
    "KXLOL",           # tournament winner
    "KXLEAGUEWORLDS",  # Worlds
]

# --- Polling cadence (seconds) ---
FAST_POLL = 5          # book poll interval for markets near/at game time
SLOW_POLL = 300        # book poll interval for other open markets
DISCOVER_EVERY = 600   # re-scan both platforms for new/updated markets
STATUS_EVERY = 60      # check Kalshi exchange status (trading halts)
TRADES_EVERY = 60      # incremental trade pull for fast-tier markets
FAST_BEFORE = 45 * 60      # start fast polling this long before game start
FAST_AFTER = 8 * 3600      # keep fast polling this long after game start

# --- Rate limiting: min seconds between requests, per host ---
MIN_INTERVAL = {
    "api.elections.kalshi.com": 0.20,
    "clob.polymarket.com": 0.05,
    "gamma-api.polymarket.com": 0.06,
    "data-api.polymarket.com": 0.05,
}
