"""Frozen, forward-only challengers on the incumbent's exact live frames.

Inference runs on a bounded in-memory queue.  No worker scans the historical
ledger for missing predictions.  A database trigger also rejects late writes,
pre-registration games, changed inputs, and predictions after known outcomes.
"""
import argparse
import copy
import datetime as dt
import hashlib
import io
import json
import logging
import os
import queue
import threading
import tempfile
import time

import numpy as np
from psycopg.types.json import Json

from . import config, util, wpgam

log = logging.getLogger("wpcandidate")
ADAPTER = "wpgam.predict_live_model:v1"
BUNDLE_KIND = "wpcandidate_pooled_bundle_v1"
MAX_PREDICTION_DELAY_S = 15
FREEZE_DIR = os.path.join(wpgam.OUT_DIR, "candidates")
SOURCE_FILES = ("wpcandidate.py", "wpgam.py", "live.py", "shadow.py",
                "wpx.py", "wpx_inputs.py", "draft.py", "wpa.py", "config.py", "util.py", "wpexposure.py", "wpdeploy.py", "wpresource.py")
_SCHEMA_READY = False
_WORKER = None


class OutcomeExposureError(RuntimeError):
    """Resolved candidate outcomes could not enter the shared exposure ledger."""

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS shadow_candidates (
        candidate_id TEXT PRIMARY KEY,
        registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        label TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        artifact_sha256 TEXT NOT NULL,
        model_kind TEXT NOT NULL,
        input_contract JSONB NOT NULL,
        input_contract_sha256 TEXT NOT NULL,
        inference_source JSONB NOT NULL,
        inference_source_sha256 TEXT NOT NULL,
        source_revision TEXT NOT NULL,
        training_cutoff TIMESTAMPTZ NOT NULL,
        evaluation_end TIMESTAMPTZ NOT NULL,
        rule JSONB NOT NULL,
        plan_sha256 TEXT NOT NULL,
        frozen_plan JSONB NOT NULL DEFAULT '{}',
        artifact_meta JSONB NOT NULL,
        CHECK (training_cutoff < registered_at),
        CHECK (evaluation_end > registered_at)
    );""",
    "ALTER TABLE shadow_candidates ADD COLUMN IF NOT EXISTS frozen_plan JSONB NOT NULL DEFAULT '{}';",
    """CREATE TABLE IF NOT EXISTS shadow_candidate_predictions (
        candidate_id TEXT NOT NULL REFERENCES shadow_candidates(candidate_id),
        protocol_id TEXT NOT NULL,
        game_id TEXT NOT NULL,
        game_start_ts BIGINT NOT NULL,
        minute INT NOT NULL,
        predicted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        candidate_p DOUBLE PRECISION CHECK (candidate_p BETWEEN 0 AND 1),
        status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
        error TEXT,
        input_state JSONB NOT NULL,
        input_sha256 TEXT NOT NULL,
        inference_source_sha256 TEXT NOT NULL,
        artifact_sha256 TEXT NOT NULL,
        prediction_metadata JSONB NOT NULL DEFAULT '{}',
        PRIMARY KEY (candidate_id, protocol_id, game_id, game_start_ts, minute),
        FOREIGN KEY (protocol_id, game_id, game_start_ts, minute)
          REFERENCES shadow_predictions(protocol_id, game_id, game_start_ts, minute),
        CHECK ((status='ok' AND candidate_p IS NOT NULL AND error IS NULL)
            OR (status='failed' AND candidate_p IS NULL AND error IS NOT NULL))
    );""",
    "ALTER TABLE shadow_candidate_predictions ADD COLUMN IF NOT EXISTS prediction_metadata JSONB NOT NULL DEFAULT '{}';",
    """CREATE OR REPLACE FUNCTION reject_shadow_candidate_mutation()
    RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'candidate registrations and forecasts are append-only'; END;
    $$;""",
    """CREATE OR REPLACE FUNCTION validate_shadow_candidate_prediction()
    RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE c shadow_candidates; p shadow_predictions;
    BEGIN
      SELECT * INTO STRICT c FROM shadow_candidates WHERE candidate_id=NEW.candidate_id;
      SELECT * INTO STRICT p FROM shadow_predictions
        WHERE protocol_id=NEW.protocol_id AND game_id=NEW.game_id
          AND game_start_ts=NEW.game_start_ts AND minute=NEW.minute;
      -- Caller-supplied timestamps cannot turn a retrospective write prospective.
      NEW.predicted_at := clock_timestamp();
      IF p.captured_at < c.registered_at
         OR to_timestamp(p.game_start_ts) < c.registered_at
         OR p.captured_at > c.evaluation_end
         OR p.feed_ts < extract(epoch FROM c.registered_at)
         OR NEW.predicted_at < p.captured_at
         OR NEW.predicted_at > p.captured_at + interval '15 seconds' THEN
        RAISE EXCEPTION 'candidate prediction is outside the prospective capture window';
      END IF;
      IF NEW.input_state IS DISTINCT FROM p.state
         OR NEW.inference_source_sha256 <> c.inference_source_sha256
         OR NEW.artifact_sha256 <> c.artifact_sha256 THEN
        RAISE EXCEPTION 'candidate prediction inputs or frozen provenance mismatch';
      END IF;
      IF EXISTS (SELECT 1 FROM shadow_outcomes
                 WHERE game_id=p.game_id AND game_start_ts=p.game_start_ts) THEN
        RAISE EXCEPTION 'candidate prediction follows a known outcome';
      END IF;
      RETURN NEW;
    END;
    $$;""",
    """DO $$ BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='shadow_candidates_immutable'
                     AND tgrelid='shadow_candidates'::regclass) THEN
        CREATE TRIGGER shadow_candidates_immutable BEFORE UPDATE OR DELETE ON shadow_candidates
          FOR EACH ROW EXECUTE FUNCTION reject_shadow_candidate_mutation();
      END IF;
      IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='shadow_candidate_predictions_immutable'
                     AND tgrelid='shadow_candidate_predictions'::regclass) THEN
        CREATE TRIGGER shadow_candidate_predictions_immutable BEFORE UPDATE OR DELETE ON shadow_candidate_predictions
          FOR EACH ROW EXECUTE FUNCTION reject_shadow_candidate_mutation();
      END IF;
      IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='shadow_candidate_predictions_prospective'
                     AND tgrelid='shadow_candidate_predictions'::regclass) THEN
        CREATE TRIGGER shadow_candidate_predictions_prospective BEFORE INSERT ON shadow_candidate_predictions
          FOR EACH ROW EXECUTE FUNCTION validate_shadow_candidate_prediction();
      END IF;
    END; $$;""",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inference_source():
    root = os.path.dirname(__file__)
    return {name: sha256(os.path.join(root, name)) for name in SOURCE_FILES}


def input_contract(model):
    """Bind the live adapter, column ordering and training timing semantics."""
    core = model["core"] if model.get("kind") == BUNDLE_KIND else model
    adapter = ("wpcandidate.predict_candidate_model:pooled_bundle_v1" if model.get("kind") == BUNDLE_KIND else
               "wpcandidate.predict_candidate_model:core_calibration_v1" if core.get("meta", {}).get("live_calibration") else ADAPTER)
    contract = {"adapter": adapter,
            "state_features": [str(n) for n in core["state"]["feature_names"]],
            "pregame_features": list(wpgam.PREGAME_FEATURES),
            "frame": "exact incumbent state with identical effective priors and draft",
            "training_input_contract": model.get("meta", {}).get("input_contract")}
    if model.get("meta", {}).get("live_calibration"):
        contract["live_calibration"] = model["meta"]["live_calibration"]
        contract["calibration_probability_clip"] = [1e-5, 1-1e-5]
    if core["state"].get("calibration_probability_clip") is not None:
        contract["native_calibration"] = {
            "intercept": float(core["state"].get("cal_intercept", 0.)),
            "slope": float(core["state"].get("cal_slope", 1.)),
            "probability_clip": list(core["state"]["calibration_probability_clip"])}
    if model.get("kind") == BUNDLE_KIND:
        contract.update(resource_features=model["resource"]["input_names"],
                        resource_mode=model["resource"]["mode"],
                        gold_players="absolute blue five role slots then red five role slots",
                        missing_gold="frozen core GAM and its independently selected calibration",
                        fallback_calibration=core["meta"].get("native_calibration") or core["meta"]["live_calibration"])
    return contract


def load_candidate_model(path):
    try:
        return wpgam.load_model(path)
    except ValueError as original:
        from . import wpresource
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["kind"].item()) != BUNDLE_KIND:
                raise original
            core = wpgam.load_model(io.BytesIO(archive["core_npz"].tobytes()))
            resource = wpresource.load(io.BytesIO(archive["resource_npz"].tobytes()))
            meta = json.loads(str(archive["meta"].item()))
        return {"kind": BUNDLE_KIND, "core": core, "resource": resource, "meta": meta}


def save_bundle(core, resource, meta, path):
    """One self-contained, pickle-free artifact; no external component paths."""
    from . import wpresource
    with tempfile.TemporaryDirectory(prefix="candidate-bundle-") as directory:
        core_path, resource_path = [os.path.join(directory, name) for name in ("core.npz", "resource.npz")]
        wpgam.save_model(core, core_path, core["meta"])
        wpresource.save(resource, resource_path)
        with open(core_path, "rb") as source:
            core_bytes = np.frombuffer(source.read(), dtype=np.uint8)
        with open(resource_path, "rb") as source:
            resource_bytes = np.frombuffer(source.read(), dtype=np.uint8)
        with open(path, "xb") as target:
            np.savez_compressed(target, kind=np.asarray(BUNDLE_KIND), core_npz=core_bytes,
                                resource_npz=resource_bytes, meta=np.asarray(canonical_json(meta)))
    return path


def apply_calibration(probabilities, calibration):
    p = np.clip(np.asarray(probabilities), 1e-5, 1-1e-5)
    if (not np.isfinite([calibration["intercept"], calibration["slope"]]).all()
            or calibration["slope"] <= 0):
        raise ValueError("Candidate calibration must be finite and monotone")
    return wpgam._sigmoid(calibration["intercept"] + calibration["slope"] * np.log(p/(1-p)))


def predict_candidate_arrays(model, raw, gold, C, t, fallback=False):
    from . import wpresource
    pooled = model.get("kind") == BUNDLE_KIND
    if pooled and not fallback:
        p = wpresource.predict(model["resource"], raw, gold, C, t, calibrated=False)
        meta = model["meta"]
    else:
        core = model["core"] if pooled else model
        p = wpgam.predict_state(core["state"], raw, t)
        meta = core.get("meta", {})
    calibration = meta.get("live_calibration")
    return apply_calibration(p, calibration) if calibration else p


def predict_candidate_model(model, state, blue_champs=(), red_champs=()):
    if model.get("kind") != BUNDLE_KIND:
        result = wpgam.predict_live_model(model, state, blue_champs, red_champs, rounded=False)
        if model.get("meta", {}).get("live_calibration"):
            result["p_blue"] = float(apply_calibration([result["p_blue"]], model["meta"]["live_calibration"])[0])
        return dict(result, candidate_path="core", fallback=False)
    core = model["core"]
    C, unknown = wpgam._live_champ_row(core["pregame"]["champ_names"], blue_champs, red_champs)
    gold = state.get("gold_players")
    try:
        gold = np.asarray(gold, dtype=float)
        complete = gold.shape == (10,) and np.isfinite(gold).all() and np.all(gold >= 0)
    except (ValueError, TypeError):
        complete = False
    if not complete:
        result = predict_candidate_model(core, state, blue_champs, red_champs)
        return dict(result, candidate_path="core_missing_gold", fallback=True)
    _, team, champ = wpgam.predict_pregame(core["pregame"], wpgam.pregame_values_from_live(state), C)
    champion_state = wpgam.champ_state_scores(core["champ_state"]["beta"], C)[0]
    raw = np.r_[core["pregame"]["intercept"]+team[0], champ[0], champion_state,
                wpgam.state_values_from_live(state)][None, :]
    p = predict_candidate_arrays(model, raw, gold[None, :], C, [float(state.get("t_min", 0) or 0)])[0]
    return {"p_blue": float(p), "candidate_path": "resource", "fallback": False, "unknown_champions": unknown}


def ensure_schema(conn):
    global _SCHEMA_READY
    if not _SCHEMA_READY:
        from . import db, shadow
        shadow._ensure_tables(conn)
        db.apply_schema(conn, SCHEMA)
        _SCHEMA_READY = True


def _datetime(value):
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=value.tzinfo or dt.timezone.utc)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, dt.timezone.utc)
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or dt.timezone.utc)


def register(conn, model_path, *, training_cutoff, evaluation_end,
             min_gain_brier, label="corrected core GAM", min_games=100,
             max_logloss_regression=0.0, precision_target=None,
             sample_size_basis=None, expected_input_contract=None,
             max_calibration_error_regression=0.0, incumbent_versions=None):
    """Freeze one artifact and a prespecified endpoint; never replace a ledger.

    min_games is an operational collection floor. A power/precision claim
    requires sample_size_basis from the corrected development series losses.
    """
    model = load_candidate_model(model_path)
    contract = input_contract(model)
    if expected_input_contract is not None and contract != expected_input_contract:
        raise ValueError("candidate input contract differs from expected contract")
    cutoff, endpoint = _datetime(training_cutoff), _datetime(evaluation_end)
    now = dt.datetime.now(dt.timezone.utc)
    if not cutoff < now < endpoint:
        raise ValueError("require training_cutoff < registration < evaluation_end")
    if min_games < 1 or not np.isfinite(min_gain_brier) or min_gain_brier < 0:
        raise ValueError("minimum games and worthwhile Brier gain must be valid")
    if precision_target is not None and (not np.isfinite(precision_target) or precision_target <= 0):
        raise ValueError("precision target must be positive")
    if not np.isfinite(max_logloss_regression):
        raise ValueError("log-loss regression threshold must be finite")
    if not np.isfinite(max_calibration_error_regression):
        raise ValueError("calibration regression threshold must be finite")
    meta = model.get("meta", {})
    embedded_cutoff = meta.get("training_cutoff")
    if embedded_cutoff is None or _datetime(embedded_cutoff) != cutoff:
        raise ValueError("artifact must contain the exact training_cutoff provenance")
    if not meta.get("input_contract"):
        raise ValueError("artifact must declare its training input_contract")
    if not meta.get("dataset_sha256"):
        raise ValueError("artifact must declare its corrected training dataset_sha256")
    artifact_sha = sha256(model_path)
    sources = inference_source()
    rule = {"version": "candidate_fixed_endpoint_v1", "prospective_only": True,
            "historical_backfill": False,
            "target_population": "all live games starting after registration; mutually valid captured model frames; no quote required",
            "primary_metric": "game-balanced Brier; candidate minus incumbent",
            "uncertainty": "paired whole normalized team-pair/UTC-date cluster bootstrap",
            "endpoint": "fixed_date", "end_date": endpoint.isoformat(),
            "metric": "game_balanced_brier", "bootstrap_unit": "series_date",
            "minimum_games": int(min_games),
            "minimum_games_role": "operational floor, not proof of adequate power",
            "min_gain_brier": float(min_gain_brier),
            "min_improvement": float(min_gain_brier),
            "max_logloss_regression": float(max_logloss_regression),
            "calibration_check": "game-balanced ten fixed-width probability bins; expected absolute calibration error",
            "max_calibration_error_regression": float(max_calibration_error_regression),
            "precision_target": precision_target, "sample_size_basis": sample_size_basis,
            "prediction_deadline_s": MAX_PREDICTION_DELAY_S,
            "decision": "one analysis after fixed endpoint; exact incumbent artifact/source cohorts; no automatic promotion"}
    from . import shadow
    incumbent_versions = incumbent_versions or shadow._artifact_versions()
    provenance = {"candidate_sha256": artifact_sha,
                  "incumbent_sha256": incumbent_versions["model_sha256"],
                  "incumbent_stack_sha256": incumbent_versions["stack_sha256"],
                  "dataset_sha256": meta["dataset_sha256"],
                  "input_contract_sha256": digest(contract),
                  "source_revision": digest(sources)}
    plan = {"artifact_sha256": artifact_sha, "input_contract_sha256": digest(contract),
            "inference_source_sha256": digest(sources),
            "inference_source": sources, "provenance": provenance,
            "training_cutoff": cutoff.isoformat(), "rule": rule}
    plan_sha = digest(plan)
    cid = "candidate-" + plan_sha
    ensure_schema(conn)
    existing = conn.execute("SELECT * FROM shadow_candidates WHERE candidate_id=%s", (cid,)).fetchone()
    if existing:
        conn.rollback()
        return dict(existing)
    os.makedirs(FREEZE_DIR, exist_ok=True)
    frozen_path = os.path.join(FREEZE_DIR, artifact_sha + ".npz")
    with open(model_path, "rb") as source:
        payload = source.read()
    if hashlib.sha256(payload).hexdigest() != artifact_sha:
        raise ValueError("candidate artifact changed while registering")
    try:
        with open(frozen_path, "xb") as target:
            target.write(payload)
        os.chmod(frozen_path, 0o444)
    except FileExistsError:
        if sha256(frozen_path) != artifact_sha:
            raise ValueError("frozen artifact path has unexpected content")
    row = conn.execute(
        """INSERT INTO shadow_candidates
          (candidate_id,label,artifact_path,artifact_sha256,model_kind,input_contract,
           input_contract_sha256,inference_source,inference_source_sha256,source_revision,
           training_cutoff,evaluation_end,rule,plan_sha256,frozen_plan,artifact_meta)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
        (cid, label, frozen_path, artifact_sha, model["kind"], Json(contract), digest(contract),
         Json(sources), digest(sources), util.source_revision(config.REPO_ROOT),
         cutoff, endpoint, Json(rule), plan_sha, Json(plan), Json(meta))).fetchone()
    conn.commit()
    return dict(row)


def _eligible(registration, capture):
    registered = _datetime(registration["registered_at"]).timestamp()
    return (registered <= capture["game_start_ts"] and registered <= capture["state"]["ts"]
            and registered <= capture["captured_at"] <= _datetime(registration["evaluation_end"]).timestamp())


def predict_frozen(registration, state, loaded_source_sha256, with_metadata=False):
    """Fail closed on a changed artifact, adapter, or imported source generation."""
    expected_source = registration["inference_source_sha256"]
    if expected_source != loaded_source_sha256 or digest(inference_source()) != expected_source:
        raise ValueError("frozen inference source mismatch; restart with the registered source")
    if sha256(registration["artifact_path"]) != registration["artifact_sha256"]:
        raise ValueError("frozen candidate artifact hash mismatch")
    model = load_candidate_model(registration["artifact_path"])
    if sha256(registration["artifact_path"]) != registration["artifact_sha256"]:
        raise ValueError("frozen candidate artifact changed while loading")
    if input_contract(model) != registration["input_contract"]:
        raise ValueError("frozen candidate input contract mismatch")
    # Outputs and capture metadata are retained as evidence, never inference inputs.
    excluded = {"p_blue", "p_blue_no_champ", "capture_timing", "priors", "priors_effective",
                "roster", "pelo_adjustment", "blend_error", "blue_champs", "red_champs"}
    inputs = {k: v for k, v in state.items()
              if k not in excluded and not k.startswith(("lo_", "model_"))}
    result = predict_candidate_model(model, inputs, state.get("blue_champs", []), state.get("red_champs", []))
    p = float(result["p_blue"])
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("candidate produced an invalid probability")
    if digest(inference_source()) != expected_source:
        raise ValueError("frozen inference source changed during prediction")
    return {"p_blue": p, "candidate_path": result["candidate_path"], "fallback": result["fallback"]} if with_metadata else p


def _write_prediction(conn, registration, capture, p, error=None):
    metadata = {}
    if isinstance(p, dict):
        metadata = {key: value for key, value in p.items() if key != "p_blue"}
        p = p["p_blue"]
    conn.execute(
        """INSERT INTO shadow_candidate_predictions
          (candidate_id,protocol_id,game_id,game_start_ts,minute,candidate_p,status,error,
           input_state,input_sha256,inference_source_sha256,artifact_sha256,prediction_metadata)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
        (registration["candidate_id"], capture["protocol_id"], capture["game_id"],
         capture["game_start_ts"], capture["minute"], p,
         "failed" if error else "ok", error, Json(capture["state"]), digest(capture["state"]),
         registration["inference_source_sha256"], registration["artifact_sha256"], Json(metadata)))
    conn.commit()


class CandidateWorker:
    """One optional worker, at most four queued live frames, no historical retries."""
    def __init__(self, connect=None, predict=None, max_queue=4):
        from . import db
        self.connect = connect or db.connect
        self.predict = predict or (lambda *args: predict_frozen(*args, with_metadata=True))
        self.queue = queue.Queue(maxsize=max_queue)
        self.loaded_source_sha256 = digest(inference_source())
        self.registrations = []
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="shadow-candidate")
        self.dropped = 0

    def start(self):
        self.thread.start()

    def submit(self, capture):
        # Registration discovery and artifact loading never run on capture's thread.
        if not self.ready.is_set() or not self.registrations:
            return False
        try:
            self.queue.put_nowait(copy.deepcopy(capture))
            return True
        except queue.Full:
            self.dropped += 1
            log.warning("candidate queue full; frame skipped, incumbent capture continues")
            return False

    def _run(self):
        conn = None
        try:
            conn = self.connect()
            ensure_schema(conn)
            self.registrations = [dict(r) for r in conn.execute(
                "SELECT * FROM shadow_candidates WHERE evaluation_end > clock_timestamp() ORDER BY registered_at").fetchall()]
            conn.rollback()
            self.ready.set()
            log.info("candidate worker loaded %d frozen registrations", len(self.registrations))
            while True:
                capture = self.queue.get()
                try:
                    for registration in self.registrations:
                        if not _eligible(registration, capture):
                            continue
                        if time.time() - capture["captured_at"] > MAX_PREDICTION_DELAY_S:
                            log.warning("candidate deadline elapsed; no historical retry")
                            continue
                        try:
                            p = self.predict(registration, capture["state"], self.loaded_source_sha256)
                            _write_prediction(conn, registration, capture, p)
                        except Exception as exc:
                            conn.rollback()
                            log.warning("candidate %s frame failed: %s", registration["candidate_id"], exc)
                            try:
                                _write_prediction(conn, registration, capture, None, str(exc)[:1000])
                            except Exception:
                                conn.rollback()
                finally:
                    self.queue.task_done()
        except Exception:
            log.exception("candidate worker unavailable; incumbent capture continues")
        finally:
            self.ready.set()
            if conn is not None:
                conn.close()


def start_worker():
    global _WORKER
    if _WORKER is None:
        _WORKER = CandidateWorker()
        _WORKER.start()
    return _WORKER


def submit_capture(capture):
    return _WORKER.submit(capture) if _WORKER is not None else False


def _cluster(row):
    from .draft import norm_team
    date = dt.datetime.fromtimestamp(row["game_start_ts"], dt.timezone.utc).date().isoformat()
    teams = sorted([norm_team(row["blue_team"]), norm_team(row["red_team"])])
    return canonical_json([date, teams])


def _cohort_key(row):
    return canonical_json([row["protocol_id"], row["stack_sha256"], row["code_revision"]])


def _paired_scores(rows, bootstrap, seed=901):
    games = {}
    for row in rows:
        key = (row["game_id"], row["game_start_ts"])
        games.setdefault(key, []).append(row)
    losses, clusters = [], []
    calibration = {"incumbent": [], "candidate": []}
    calibration_bins = {name: np.zeros((10, 3)) for name in calibration}
    for frame_rows in games.values():
        y = np.asarray([r["blue_win"] for r in frame_rows], dtype=float)
        p = np.asarray([[r["model_p"], r["candidate_p"]] for r in frame_rows], dtype=float)
        p_clip = np.clip(p, 1e-7, 1 - 1e-7)
        losses.append(np.r_[np.mean((p-y[:, None])**2, axis=0),
                            np.mean(-(y[:, None]*np.log(p_clip)+(1-y[:, None])*np.log(1-p_clip)), axis=0)])
        clusters.append(_cluster(frame_rows[0]))
        for i, name in enumerate(calibration):
            calibration[name].append([float(np.mean(p[:, i])), float(np.mean(y)),
                                      float(np.mean(p[:, i]-y))])
            bins = np.minimum((p[:, i]*10).astype(int), 9)
            for bin_id in range(10):
                mask = bins == bin_id
                # Each game contributes total weight one, however many frames.
                calibration_bins[name][bin_id] += [float(np.sum(mask))/len(y),
                                                   float(np.sum(p[mask, i]))/len(y),
                                                   float(np.sum(y[mask]))/len(y)]
    loss = np.asarray(losses)
    delta = loss[:, 1] - loss[:, 0]
    unique = sorted(set(clusters))
    groups = [np.flatnonzero(np.asarray(clusters) == key) for key in unique]
    ci = None
    if bootstrap and len(groups) >= 2:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap):
            ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            draws.append(float(np.mean(delta[ix])))
        ci = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    means = loss.mean(axis=0)
    calibrated = {}
    for name, values in calibration.items():
        mean = np.mean(values, axis=0)
        bins = calibration_bins[name]
        calibrated[name] = {"mean_probability": float(mean[0]), "outcome_rate": float(mean[1]),
                            "mean_residual": float(mean[2]),
                            "expected_calibration_error": float(np.abs(bins[:, 1]-bins[:, 2]).sum()/len(games)),
                            "bins": [{"lo": i/10, "hi": (i+1)/10,
                                      "game_weight": float(n/len(games)),
                                      "predicted": float(p/n) if n else None,
                                      "observed": float(y/n) if n else None}
                                     for i, (n, p, y) in enumerate(bins)]}
    return {"games": len(games), "states": len(rows), "series_date_clusters": len(groups),
            "scores": {"incumbent": {"brier_game": float(means[0]), "logloss_game": float(means[2])},
                       "candidate": {"brier_game": float(means[1]), "logloss_game": float(means[3])}},
            "candidate_minus_incumbent": float(delta.mean()), "ci95": ci,
            "logloss_delta": float(means[3]-means[2]),
            "calibration": calibrated,
            "bootstrap_unit": "series_date",
            "cluster_definition": "normalized team pair and UTC start date; all maps retained together"}


def score_rows(registration, rows, bootstrap=5000, now=None):
    """One frozen rule, exact incumbent artifact/source cohorts, paired frames."""
    now = _datetime(now or dt.datetime.now(dt.timezone.utc))
    endpoint_complete = now >= _datetime(registration["evaluation_end"])
    valid = [r for r in rows if r.get("status") == "ok" and r.get("blue_win") in (0, 1)
             and r.get("candidate_p") is not None and r.get("stack_sha256")
             and r.get("code_revision") and r.get("input_sha256") == digest(r["state"])]
    cohorts = {}
    for key in sorted({_cohort_key(r) for r in valid}):
        selected = [r for r in valid if _cohort_key(r) == key]
        result = _paired_scores(selected, bootstrap)
        result.update(protocol_id=selected[0]["protocol_id"], stack_sha256=selected[0]["stack_sha256"],
                      code_revision=selected[0]["code_revision"], endpoint_complete=endpoint_complete)
        rule = registration["rule"]
        result["operational_floor_met"] = result["games"] >= rule["minimum_games"]
        precision = rule.get("precision_target")
        result["precision_met"] = (result["ci95"] is not None and precision is not None
                                    and (result["ci95"][1]-result["ci95"][0])/2 <= precision)
        result["calibration_error_delta"] = (
            result["calibration"]["candidate"]["expected_calibration_error"]
            - result["calibration"]["incumbent"]["expected_calibration_error"])
        result["passed"] = bool(endpoint_complete and result["operational_floor_met"]
                                 and result["precision_met"] and result["ci95"][1] < -rule["min_gain_brier"]
                                 and result["logloss_delta"] <= rule["max_logloss_regression"]
                                 and result["calibration_error_delta"] <= rule.get("max_calibration_error_regression", 0.0))
        result["interpretation"] = "fixed-endpoint result" if endpoint_complete else "descriptive interim; no promotion decision"
        plan = registration.get("frozen_plan", {})
        provenance = plan.get("provenance", {})
        if (provenance and selected[0].get("model_sha256") == provenance.get("incumbent_sha256")
                and selected[0]["stack_sha256"] == provenance.get("incumbent_stack_sha256")):
            result["direct_evidence"] = dict(
                provenance, kind="wpx_incumbent_comparison_v1", frozen_plan=plan,
                plan_sha256=registration["plan_sha256"], inference_source=plan["inference_source"],
                bootstrap_unit="series_date", game_balanced=True,
                endpoint_complete=endpoint_complete, games=result["games"],
                clusters=result["series_date_clusters"], ci95=result["ci95"],
                candidate_minus_incumbent=result["candidate_minus_incumbent"],
                candidate_minus_incumbent_logloss=result["logloss_delta"],
                candidate_minus_incumbent_calibration_error=result["calibration_error_delta"],
                passed=result["passed"])
        cohorts[hashlib.sha256(key.encode()).hexdigest()] = result
    return {"candidate_id": registration["candidate_id"], "artifact_sha256": registration["artifact_sha256"],
            "input_contract_sha256": registration["input_contract_sha256"],
            "inference_source_sha256": registration["inference_source_sha256"],
            "source_revision": registration["source_revision"], "plan_sha256": registration["plan_sha256"],
            "training_cutoff": str(registration["training_cutoff"]), "rule": registration["rule"],
            "endpoint_complete": endpoint_complete, "eligible_incumbent_rows": len(rows),
            "candidate_rows": sum(r.get("candidate_p") is not None for r in rows),
            "failed_rows": sum(r.get("status") == "failed" for r in rows),
            "fallback_rows": sum(bool((r.get("prediction_metadata") or {}).get("fallback")) for r in rows),
            "missing_rows": sum(r.get("status") is None for r in rows),
            "resolved_paired_rows": len(valid), "market_quote_required": False,
            "by_incumbent_version": cohorts}


def score(conn, candidate_id=None, bootstrap=5000):
    ensure_schema(conn)
    registrations = conn.execute("SELECT * FROM shadow_candidates WHERE (%s::text IS NULL OR candidate_id=%s) ORDER BY registered_at",
                                 (candidate_id, candidate_id)).fetchall()
    if candidate_id and not registrations:
        raise ValueError("unknown candidate %s" % candidate_id)
    reports = []
    from . import shadow, wpexposure
    has_feed_games = shadow._table_exists(conn, "feed_games")
    for registration in registrations:
        rows = conn.execute(
            """SELECT p.*, cp.candidate_p, cp.status, cp.error, cp.input_sha256, cp.prediction_metadata, o.blue_win
               FROM shadow_predictions p
               LEFT JOIN shadow_candidate_predictions cp
                 ON cp.candidate_id=%s AND cp.protocol_id=p.protocol_id AND cp.game_id=p.game_id
                 AND cp.game_start_ts=p.game_start_ts AND cp.minute=p.minute
               LEFT JOIN shadow_outcomes o ON o.game_id=p.game_id AND o.game_start_ts=p.game_start_ts
                 AND o.status='resolved'
               WHERE p.captured_at >= %s AND p.captured_at <= %s
                 AND to_timestamp(p.game_start_ts) >= %s
                 AND to_timestamp(p.feed_ts) >= %s
               ORDER BY p.captured_at""",
            (registration["candidate_id"], registration["registered_at"], registration["evaluation_end"],
             registration["registered_at"], registration["registered_at"])).fetchall()
        rows = [dict(r) for r in rows]
        # Reading a resolved forward outcome consumes it even if candidate
        # inference failed. Preserve exposure before publishing diagnostics.
        resolved = {(r["game_id"], r["game_start_ts"]): r for r in rows
                    if r.get("blue_win") in (0, 1)}
        exposure = None
        if resolved:
            feed_ids = [game for game, _ in resolved]
            try:
                mapping = ({str(r["esports_game_id"]): r["golgg_game_id"] for r in conn.execute(
                    "SELECT esports_game_id,golgg_game_id FROM feed_games WHERE esports_game_id=ANY(%s)",
                    (feed_ids,)).fetchall()} if has_feed_games else {})
                exposure = wpexposure.expose(
                    [mapping.get(game) for game, _ in resolved],
                    [dt.datetime.fromtimestamp(start, dt.timezone.utc).date().isoformat() for _, start in resolved],
                    registration["candidate_id"], registration["plan_sha256"], feed_game_ids=feed_ids)
            except Exception as exc:
                raise OutcomeExposureError("candidate outcomes could not be safely recorded as exposed") from exc
        report = score_rows(dict(registration), rows, bootstrap)
        report["outcome_exposure"] = exposure
        reports.append(report)
    conn.rollback()
    return {"scored_at": dt.datetime.now(dt.timezone.utc).isoformat(), "candidates": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("model_path")
    reg.add_argument("--label", default="corrected core GAM")
    reg.add_argument("--training-cutoff", required=True)
    reg.add_argument("--evaluation-end", required=True)
    reg.add_argument("--min-gain-brier", type=float, required=True)
    reg.add_argument("--min-games", type=int, default=100)
    reg.add_argument("--precision-target", type=float, required=True)
    reg.add_argument("--sample-size-basis", help="JSON file with corrected series-loss pilot calculation", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--candidate-id")
    scoring.add_argument("--bootstrap", type=int, default=5000)
    sub.add_parser("status")
    args = parser.parse_args()
    from . import db
    with db.connect() as conn:
        if args.command == "register":
            with open(args.sample_size_basis) as source:
                basis = json.load(source)
            result = register(conn, args.model_path, training_cutoff=args.training_cutoff,
                              evaluation_end=args.evaluation_end, min_gain_brier=args.min_gain_brier,
                              label=args.label, min_games=args.min_games, precision_target=args.precision_target,
                              sample_size_basis=basis)
        else:
            result = score(conn, getattr(args, "candidate_id", None), getattr(args, "bootstrap", 0))
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
