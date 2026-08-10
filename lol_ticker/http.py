"""Minimal stdlib HTTP client with per-host rate limiting and retries.

Thread-safe: request *starts* are spaced per host at the configured minimum
interval, but responses may overlap, so a worker pool reaches the full rate
budget instead of one-request-per-round-trip.
"""
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

_next_slot = {}  # host -> earliest monotonic time the next request may start
_slot_lock = threading.Lock()

SHUTDOWN = threading.Event()  # set to abort waits/retries during teardown

USER_AGENT = "lol-ticker-collector/1.0"


class ShuttingDown(Exception):
    pass


class HttpError(Exception):
    def __init__(self, status, url, body=""):
        self.status = status
        self.url = url
        self.body = body[:500]
        super().__init__("HTTP %s for %s: %s" % (status, url, self.body))


_penalty = {}  # host -> current 429 cooldown seconds


def _throttle(url):
    host = urllib.parse.urlparse(url).netloc
    min_iv = config.MIN_INTERVAL.get(host, 0.1)
    with _slot_lock:
        now = time.monotonic()
        start = max(now, _next_slot.get(host, now))
        _next_slot[host] = start + min_iv
    if start > now and SHUTDOWN.wait(start - now):
        raise ShuttingDown()
    return host


def _rate_limited(host):
    """A 429 arrived: push the whole host's schedule back, growing each time."""
    with _slot_lock:
        pen = min(_penalty.get(host, 1.0) * 2, 60.0)
        _penalty[host] = pen
        _next_slot[host] = max(_next_slot.get(host, 0), time.monotonic() + pen)


def _rate_ok(host):
    with _slot_lock:
        _penalty.pop(host, None)


def request_json(url, params=None, body=None, retries=4, timeout=30):
    """GET (or POST when body is not None) a JSON endpoint."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    last_err = None
    for attempt in range(retries + 1):
        host = _throttle(url)
        req = urllib.request.Request(url, data=data, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if data else {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                _rate_ok(host)
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode(errors="replace")
            except Exception:
                pass
            last_err = HttpError(e.code, url, body_text)
            if e.code == 429:
                _rate_limited(host)  # cool the whole host down, not just this call
            elif 400 <= e.code < 500:
                raise last_err  # other 4xx will not succeed on retry
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            last_err = e
        if attempt < retries and SHUTDOWN.wait(2 ** attempt):
            raise ShuttingDown()
    raise last_err


def get_json(url, params=None, **kw):
    return request_json(url, params=params, **kw)


def post_json(url, body, **kw):
    return request_json(url, body=body, **kw)
