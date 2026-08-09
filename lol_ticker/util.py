import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

_ISO_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?"
    r"(Z|[+-]\d{2}(?::?\d{2})?)?"
)


def parse_ts(value):
    """Parse an ISO-ish datetime string (or epoch number) to unix seconds, else None.

    Tolerates 'Z', '+00', '+0000', '+00:00' suffixes and a space separator
    (Python 3.9's fromisoformat can't).  Naive strings are assumed UTC.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        v = float(value)
        return int(v / 1000) if v > 1e12 else int(v)
    m = _ISO_RE.match(str(value).strip())
    if not m:
        return None
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    frac = m.group(7)
    us = int(float("0." + frac) * 1e6) if frac else 0
    dt = datetime(y, mo, d, h, mi, s, us, tzinfo=timezone.utc)
    tz = m.group(8)
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        digits = tz[1:].replace(":", "")
        off_h = int(digits[:2])
        off_m = int(digits[2:4]) if len(digits) >= 4 else 0
        dt -= timedelta(seconds=sign * (off_h * 3600 + off_m * 60))
    return int(dt.timestamp())


def book_hash(bids, asks):
    payload = json.dumps([bids, asks], separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def trade_hash(*fields):
    payload = "|".join(str(f) for f in fields)
    return hashlib.sha1(payload.encode()).hexdigest()
