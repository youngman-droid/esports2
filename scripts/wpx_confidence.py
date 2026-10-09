"""Save the fixed confidence candidate and audit historical evaluability.

The corrected state archive stores rating differences, not the certified
per-side source dates, rosters and support needed by this candidate. Report
that limitation explicitly instead of manufacturing retrospective confidence.
No outcomes are scored, database queried, or production artifact changed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lol_ticker import wpconfidence, wpgam


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(dataset, output):
    if output.exists():
        raise FileExistsError("Use a new output directory")
    manifest_path = Path(str(dataset)+".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["sha256"] != sha(dataset) or manifest.get("zero_future_timestamp_joins") is not True:
        raise ValueError("Hash-matched corrected archive required")
    with np.load(dataset, allow_pickle=False) as archive:
        if max(archive["date"].astype(str)) > "2026-09-02":
            raise ValueError("This metadata audit is restricted through September 2")
        columns = archive["names"].tolist()
        available_keys = archive.files
        fixed_rows = int(np.sum(archive["seq"] < 0))
    required = ["prior_provenance", "prior_availability", "roster", "pelo_adjustment",
                "elo_blue", "elo_red", "pelo_blue", "pelo_red"]
    missing = [k for k in required if k not in available_keys and k not in columns]
    output.mkdir(parents=True)
    policy = dict(kind=wpconfidence.KIND, spec=dict(wpconfidence.DEFAULT_SPEC),
                  input_contract=dict(wpconfidence.INPUT_CONTRACT), learned=False,
                  production_applied=False)
    (output/"candidate_policy.json").write_text(json.dumps(policy, indent=2)+"\n")
    source = Path(wpconfidence.__file__)
    (output/"wpconfidence.py").write_bytes(source.read_bytes())
    (output/"wpx_confidence.py").write_bytes(Path(__file__).read_bytes())
    report = dict(completed=True, metadata_audit_only=True, historical_accuracy_evaluated=False,
        reason="missing_certified_per_side_provenance" if missing else "metadata_present_requires_source_audit",
        missing_inputs=missing, dataset_sha256=sha(dataset), input_manifest_sha256=sha(manifest_path),
        source_sha256=sha(source), policy_sha256=sha(output/"candidate_policy.json"),
        games=manifest["games"], states=fixed_rows,
        source_support="live provenance may supply recent-window games; player records supply past last_ts/games",
        limitations=["fixed shrink policy is an unvalidated hypothesis, not a measured accuracy gain",
            "exp_diff is a difference of log sample counts; cannot recover two counts or source rosters",
            "day-only rating dates are conservative UTC-day proxies, not precise source times"],
        production_changed=False, prospective_registration=False)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"completion.json").write_text(json.dumps(dict(completed=True,
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()}, production_changed=False), indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(wpgam.OUT_DIR)/"states_inputs_v2_canonical_before_2026-09-03.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); main(args.dataset, args.out)
