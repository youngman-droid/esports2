"""Minimal stdlib HTTP client with per-host rate limiting and retries."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

_last_request = {}  # host -> monotonic time of last request

USER_AGENT = "lol-ticker-collector/1.0"


class HttpError(Exception):
    def __init__(self, status, url, body=""):
        self.status = status
        self.url = url
        self.body = body[:500]
        super().__init__("HTTP %s for %s: %s" % (status, url, self.body))


def _throttle(url):
    host = urllib.parse.urlparse(url).netloc
    min_iv = config.MIN_INTERVAL.get(host, 0.1)
    last = _last_request.get(host)
    now = time.monotonic()
    if last is not None and now - last < min_iv:
        time.sleep(min_iv - (now - last))
    _last_request[host] = time.monotonic()


def request_json(url, params=None, body=None, retries=4, timeout=30):
    """GET (or POST when body is not None) a JSON endpoint."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    last_err = None
    for attempt in range(retries + 1):
        _throttle(url)
        req = urllib.request.Request(url, data=data, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if data else {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode(errors="replace")
            except Exception:
                pass
            last_err = HttpError(e.code, url, body_text)
            # 4xx other than 429 will not succeed on retry
            if e.code != 429 and 400 <= e.code < 500:
                raise last_err
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            last_err = e
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise last_err


def get_json(url, params=None, **kw):
    return request_json(url, params=params, **kw)


def post_json(url, body, **kw):
    return request_json(url, body=body, **kw)
