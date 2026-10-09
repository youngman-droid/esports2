"""Recover the exact pre-May-2026 v8 artifact behind the postdraft replay.

The original run saved predictions but not its fitted incumbent. This runner
pins its archived inputs, verifies every saved held-out prediction, and writes
only an isolated analysis artifact. It never changes deployment or exposure.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

# Must precede NumPy/SciPy imports; recovery shares this machine with other work.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lol_ticker import wpx

LOG = logging.getLogger("recover_v8")
PINNED = {
    "dataset": ("data/refresh_20260912/states.npz",
                "408b0e7b37ff6402bcbbe14a21da5b05314cbe8da4770037a1d9716a3413de4c"),
    "source": ("data/wpx/repairs_v9/incumbent_source.py",
               "64c88edafb216a9e169def396f000a0f8f75a6ab816fb30f3ad00ccc28a4d7dc"),
    "predictions": ("data/wpx/repairs_v9/frozen_predictions.npz",
                    "38d2794ad27bbf1b1bbb1ada9f995a624c07319d533f471efd2ad9e995020060"),
}
INTERACTED = {"t", "t2", "gold_k_x_t", "d_kill_x_t", "dead_diff_x_t",
              "lead_x_inhib", "baron_up_x_dead", "rel_lead"}
DEFAULT_OUTPUT = ROOT / "data/wpx/postdraft_lpl_2026-09-12"
CUTOFF = "2026-05-01"
TOLERANCE = 1e-10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError("Hash mismatch: %s (expected %s, got %s)" % (path, expected, actual))
    return actual


def validate_predictions(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Prediction shape or finite-value mismatch")
    delta = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=TOLERANCE, rtol=0):
        raise ValueError("Frozen predictions mismatch: maximum absolute difference %.17g" % delta)
    return delta


def load_source(path):
    spec = importlib.util.spec_from_file_location("lol_ticker._recovered_v8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MODEL_KIND != "wpgam_v8_recency_series":
        raise ValueError("Unexpected recorded model kind")
    return module


def identity():
    inputs = {}
    for label, (relative, expected) in PINNED.items():
        path = ROOT / relative
        inputs[label] = dict(path=relative, sha256=require_hash(path, expected))
    if set(wpx.ALREADY_INTERACTED) != INTERACTED:
        raise ValueError("Champion-state interaction dependency has changed")
    return dict(inputs=inputs, implementation_sha256=sha(__file__),
                already_interacted=sorted(INTERACTED), training_cutoff_exclusive=CUTOFF,
                prediction_tolerance_absolute=TOLERANCE, prediction_tolerance_relative=0,
                numpy_version=np.__version__, scipy_version=scipy.__version__, blas_threads=4)


def main(output=DEFAULT_OUTPUT):
    output = Path(output)
    provenance = identity()
    model_path, report_path = output / "recovered_v8.npz", output / "recovery.json"
    if model_path.exists() or report_path.exists():
        if not (model_path.exists() and report_path.exists()):
            raise ValueError("Incomplete recovery outputs; refusing to overwrite")
        report = json.loads(report_path.read_text())
        if report.get("identity") != provenance or report.get("accepted") is not True:
            raise ValueError("Recovery provenance changed or existing recovery was not accepted")
        require_hash(model_path, report["artifact_sha256"])
        LOG.info("Verified existing recovered artifact %s", model_path)
        return report

    old = load_source(ROOT / PINNED["source"][0])
    started = time.monotonic()
    with np.load(ROOT / PINNED["dataset"][0], allow_pickle=False) as archive:
        # Materialize once: NpzFile otherwise repeatedly decompresses large arrays.
        arrays = {key: archive[key] for key in ("X", "y", "gid", "t", "seq", "C", "date",
                                               "names", "champ_names")}
        first, train_games, method = old._date_split(archive)
    dates = arrays["date"][first].astype(str)
    if method != "date" or not np.array_equal(train_games, dates < CUTOFF):
        raise ValueError("Archived outer split no longer matches the pre-May protocol")
    test = (arrays["seq"] < 0) & (arrays["date"] >= CUTOFF)
    with np.load(ROOT / PINNED["predictions"][0], allow_pickle=False) as frozen:
        expected = frozen["v8"]
        if (int(test.sum()) != 99408 or not np.array_equal(arrays["gid"][test], frozen["gid"])
                or not np.array_equal(arrays["y"][test], frozen["y"])):
            raise ValueError("Archived held-out rows differ from the frozen replay")
    LOG.info("Recovering from %d games strictly before %s; verifying %d held-out rows",
             train_games.sum(), CUTOFF, test.sum())
    with threadpool_limits(limits=4):
        model = old.fit_arrays(arrays["X"], arrays["y"], arrays["gid"], arrays["t"],
            arrays["seq"], arrays["C"], list(arrays["names"]), list(arrays["champ_names"]),
            train_games=train_games, dates=arrays["date"])
        X, C, names = arrays["X"][test], arrays["C"][test], list(arrays["names"])
        raw = np.column_stack([old.prior_values_from_matrix(model, X, names, C),
                               old.state_values_from_matrix(X, names)])
        times = arrays["t"][test] / 60.0
        actual = old.predict_state(model["state"], raw, times)
        delta = validate_predictions(actual, expected)
    report = dict(identity=provenance, accepted=True,
        train_games=int(train_games.sum()), train_states=int(model["train_states"]),
        fit_max_date=str(max(dates[train_games])), test_games=int((~train_games).sum()),
        verified_prediction_rows=int(test.sum()), max_absolute_prediction_difference=delta,
        exact_prediction_equality=bool(np.array_equal(actual, expected)),
        fit_seconds=time.monotonic() - started,
        production_changed=False, evaluation_registry_changed=False,
        historical_caveat="Original v8 training-input and calibration limitations are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "recovered_v8.pending.npz"
    if pending.exists():
        raise ValueError("Pending recovery artifact exists; refusing to overwrite")
    old.save_model(model, str(pending), meta=report)
    restored = old.load_model(str(pending))
    restored_p = old.predict_state(restored["state"], raw, times)
    validate_predictions(restored_p, actual)
    pending.rename(model_path)
    report["artifact_sha256"] = sha(model_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    LOG.info("Accepted recovery: max prediction difference %.17g; %s", delta, model_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.output)
