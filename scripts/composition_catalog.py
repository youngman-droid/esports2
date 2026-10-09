"""Fetch immutable champion catalogs and preserve conservative source timing.

Only static Riot Data Dragon files are fetched; no games/outcomes or paid API.
Last-Modified certifies when the returned representation was available. If a
server cannot supply it, an explicit documented source availability timestamp
is required. Retrospective catalogs whose header is later than a draft remain
unavailable for that draft rather than inventing an earlier release date.
"""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpcomposition as composition


def fetch(version, root, *, opener=urllib.request.urlopen, available_from_ts=None):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("An exact Data Dragon release is required")
    root = Path(root)
    catalog_path, raw_path = root / "catalogs" / (version + ".json"), root / "raw" / (version + ".json")
    if catalog_path.exists() or raw_path.exists():
        if not catalog_path.exists() or not raw_path.exists():
            raise ValueError("Partial catalog snapshot exists; inspect it before retrying")
        wrapper = composition.load_catalog(catalog_path)
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != wrapper.get("source_sha256"):
            raise ValueError("Saved raw catalog differs from its source hash")
        return wrapper
    url = "https://ddragon.leagueoflegends.com/cdn/%s/data/en_US/champion.json" % version
    with opener(url, timeout=30) as response:
        raw = response.read()
        modified = response.headers.get("Last-Modified")
        if available_from_ts is None:
            if not modified:
                raise ValueError("Source Last-Modified missing; documented availability timestamp required")
            available_from_ts = parsedate_to_datetime(modified).timestamp()
        payload = json.loads(raw)
        if payload.get("version") != version:
            raise ValueError("Data Dragon returned a different release")
        wrapper = composition.catalog_from_payload(payload, source_url=url,
            available_from_ts=float(available_from_ts), raw_sha256=hashlib.sha256(raw).hexdigest())
        wrapper.update(retrieved_utc=datetime.now(timezone.utc).isoformat(),
                       source_last_modified=modified, timing_source="http_last_modified" if modified else "explicit_source_evidence")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    catalog_path.write_text(json.dumps(wrapper, sort_keys=True, indent=2))
    return wrapper


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--versions", nargs="+")
    group.add_argument("--consumed-input", help="Saved already-consumed draft input through September 2, 2026 only")
    parser.add_argument("--out-root", default="data/composition")
    args = parser.parse_args()
    versions = args.versions
    if args.consumed_input:
        games = json.loads(Path(args.consumed_input).read_text())
        if any(not isinstance(g.get("day"), str) or g["day"] >= "2026-09-03" for g in games):
            raise ValueError("Saved inputs extend beyond the consumed-outcome cutoff")
        versions = sorted({".".join(str(int(x)) for x in g["patch"].split(".")) + ".1"
                           for g in games if re.fullmatch(r"\d+\.\d+", g.get("patch") or "")},
                          key=lambda p: tuple(map(int, p.split("."))))
    for version in versions:
        result = fetch(version, args.out_root)
        print(json.dumps(dict(version=version, champions=len(result["data"]),
                              available_from_ts=result["available_from_ts"],
                              source_sha256=result["source_sha256"])), flush=True)


if __name__ == "__main__":
    main()
