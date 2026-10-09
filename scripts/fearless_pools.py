"""Build an offline observed-pool snapshot from consumed Oracle's Elixir picks.

The archive has UTC dates, so the appearance timing upper bound is next-day
midnight. Strict cutoff gating drops boundary-day ambiguity. IDs remain in the
Oracle's Elixir namespace; this file does not claim a Riot esports-ID bridge.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpfearless as fearless


def build(games, as_of_date):
    if any(not isinstance(g.get("day"), str) or g["day"] >= "2026-09-03" for g in games):
        raise ValueError("Only already-consumed appearances through September 2, 2026 may enter")
    cutoff = datetime.fromisoformat(as_of_date).replace(tzinfo=timezone.utc).timestamp()
    records = []
    for game in games:
        upper = (datetime.fromisoformat(game["day"]).replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp()
        for side in ("blue", "red"):
            for role in fearless.ROLES:
                player, champion = game[side]["players"].get(role), game[side]["picks"].get(role)
                if player and champion:
                    records.append(dict(game_id=game["id"], player_id=str(player), role=role,
                                        champion=champion, completed_ts=upper))
    snapshot = fearless.build_player_pools(records, as_of_ts=cutoff)
    snapshot.update(identity_namespace="oracle_elixir_player_id",
                    timing="earlier UTC date appearance upper bound; not an asserted exact completion clock",
                    max_input_date=max(g["day"] for g in games),
                    live_identity_bridge_available=False)
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/wpx/draft_comfort_comparison_20260916/inputs.json")
    parser.add_argument("--as-of-date", default="2026-09-03")
    parser.add_argument("--out", default="data/fearless/offline_player_pools_2026-09-03.json")
    args = parser.parse_args()
    path = Path(args.input)
    snapshot = build(json.loads(path.read_text()), args.as_of_date)
    snapshot["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError("Preserve existing pool snapshots; select a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, sort_keys=True, indent=2))
    print(json.dumps({key: snapshot[key] for key in ("records", "rejected_records", "excluded_records",
        "identity_namespace", "max_input_date", "live_identity_bridge_available")}, indent=2))


if __name__ == "__main__":
    main()
