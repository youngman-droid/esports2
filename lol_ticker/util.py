import hashlib
import json
import os
import re
import subprocess
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


def source_revision(repo_root):
    """Git revision plus a deterministic fingerprint for dirty source trees."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = "unknown", True
    if not dirty:
        return head
    digest = hashlib.sha256()
    try:
        digest.update(subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"], cwd=repo_root,
            stderr=subprocess.DEVNULL, timeout=10))
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_root, stderr=subprocess.DEVNULL, timeout=5)
        for rel_raw in sorted(x for x in untracked.split(b"\0") if x):
            rel = rel_raw.decode(errors="surrogateescape")
            digest.update(rel_raw + b"\0")
            with open(os.path.join(repo_root, rel), "rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(block)
    except (OSError, subprocess.SubprocessError):
        digest.update(b"unavailable-dirty-source")
    return "%s+worktree.%s" % (head, digest.hexdigest()[:16])
