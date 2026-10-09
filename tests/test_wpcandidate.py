import copy
import datetime as dt
import os
import tempfile
import threading
import time
import unittest
import uuid
from unittest import mock

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

from lol_ticker import config, wpcandidate as candidate


class CandidateInferenceTests(unittest.TestCase):
    def test_exact_frame_and_effective_priors_reach_adapter_without_predictions(self):
        model = {"state": {"feature_names": ["gold_k"]}, "meta": {"input_contract": "causal"}}
        contract = candidate.input_contract(model)
        state = {"ts": 1000, "gold_blue": 4200, "elo_diff": 88, "priors_effective": {"elo_diff": 88},
                 "blue_champs": ["Ahri"], "red_champs": ["Orianna"],
                 "p_blue": .99, "lo_state": 9, "capture_timing": {"a": 1}}
        source = {"wpgam.py": "frozen"}
        registration = {"inference_source_sha256": candidate.digest(source),
                        "artifact_path": "frozen.npz", "artifact_sha256": "artifact",
                        "input_contract": contract}
        with mock.patch.object(candidate, "inference_source", return_value=source), \
                mock.patch.object(candidate, "sha256", return_value="artifact"), \
                mock.patch.object(candidate.wpgam, "load_model", return_value=model), \
                mock.patch.object(candidate.wpgam, "predict_live_model", return_value={"p_blue": .625}) as predict:
            self.assertEqual(candidate.predict_frozen(registration, state, candidate.digest(source)), .625)
            args, kwargs = predict.call_args
            self.assertEqual(args[1], {"ts": 1000, "gold_blue": 4200, "elo_diff": 88})
            self.assertEqual(args[2:], (["Ahri"], ["Orianna"]))
            self.assertFalse(kwargs["rounded"])
            changed = dict(registration, artifact_sha256="different")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                candidate.predict_frozen(changed, state, candidate.digest(source))
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                candidate.predict_frozen(registration, state, "stale-imported-source")
            changed = dict(registration, input_contract={"adapter": "unknown"})
            with self.assertRaisesRegex(ValueError, "input contract mismatch"):
                candidate.predict_frozen(changed, state, candidate.digest(source))

    def test_bounded_submission_is_nonblocking_and_copies_input(self):
        worker = candidate.CandidateWorker(max_queue=1)
        worker.ready.set()
        worker.registrations = [{"candidate_id": "one"}]
        capture = {"state": {"gold_blue": 10}}
        self.assertTrue(worker.submit(capture))
        capture["state"]["gold_blue"] = 500
        self.assertFalse(worker.submit(capture))
        self.assertEqual(worker.queue.get_nowait()["state"]["gold_blue"], 10)
        self.assertEqual(worker.dropped, 1)
        self.assertFalse(worker.thread.is_alive())

    def test_slow_optional_inference_does_not_block_new_capture_submission(self):
        entered, release = threading.Event(), threading.Event()
        now = time.time()
        reg = {"candidate_id": "frozen", "registered_at": now-1000, "evaluation_end": now+1000}
        conn = mock.MagicMock()
        conn.execute.return_value.fetchall.return_value = [reg]

        def slow_predict(*args):
            entered.set()
            release.wait(2)
            return .6

        worker = candidate.CandidateWorker(connect=lambda: conn, predict=slow_predict, max_queue=1)
        frame = {"game_start_ts": now-100, "captured_at": now, "state": {"ts": now-10}}
        with mock.patch.object(candidate, "ensure_schema"), \
                mock.patch.object(candidate, "_write_prediction"):
            worker.start()
            self.assertTrue(worker.ready.wait(1))
            self.assertTrue(worker.submit(frame))
            self.assertTrue(entered.wait(1))
            began = time.monotonic()
            self.assertTrue(worker.submit(frame))
            self.assertFalse(worker.submit(frame))
            self.assertLess(time.monotonic()-began, .1)
            release.set()
            worker.queue.join()

    def test_pre_registration_games_never_enter_a_prospective_cohort(self):
        reg = {"registered_at": 100, "evaluation_end": 500}
        capture = {"game_start_ts": 99, "captured_at": 105, "state": {"ts": 101}}
        self.assertFalse(candidate._eligible(reg, capture))
        capture["game_start_ts"] = 100
        self.assertTrue(candidate._eligible(reg, capture))
        capture["state"]["ts"] = 99
        self.assertFalse(candidate._eligible(reg, capture))


class CandidateScoreTests(unittest.TestCase):
    @staticmethod
    def registration():
        return {"candidate_id": "frozen", "artifact_sha256": "artifact", "input_contract_sha256": "contract",
                "inference_source_sha256": "source", "source_revision": "git", "plan_sha256": "plan",
                "training_cutoff": 0, "evaluation_end": 2000000000,
                "rule": {"minimum_games": 2, "min_gain_brier": .001, "precision_target": .1,
                         "max_logloss_regression": 0}}

    @staticmethod
    def row(game, minute=1, p=.7, incumbent=.5, outcome=1, start=1900000000):
        state = {"ts": start + minute * 60, "p_blue": incumbent, "gold_blue": 10000}
        return {"game_id": game, "game_start_ts": start, "minute": minute,
                "protocol_id": "v12", "stack_sha256": "v8", "code_revision": "code1",
                "status": "ok", "model_p": incumbent, "candidate_p": p, "blue_win": outcome,
                "blue_team": "Alpha", "red_team": "Beta", "state": state,
                "input_sha256": candidate.digest(state), "polymarket_p": None, "kalshi_p": None}

    def test_model_comparison_uses_no_quotes_and_balances_games(self):
        rows = [self.row("g1", 1, p=.9), self.row("g1", 2, p=.9),
                self.row("g2", 1, p=.7, start=1900086400)]
        got = candidate.score_rows(self.registration(), rows, bootstrap=100, now=2000000001)
        score = next(iter(got["by_incumbent_version"].values()))
        self.assertEqual((score["games"], score["states"]), (2, 3))
        self.assertAlmostEqual(score["scores"]["candidate"]["brier_game"], .05)
        self.assertAlmostEqual(score["candidate_minus_incumbent"], -.2)
        self.assertEqual(score["series_date_clusters"], 2)
        self.assertTrue(score["passed"])
        self.assertFalse(got["market_quote_required"])

    def test_multiple_maps_same_pair_date_stay_in_one_cluster(self):
        rows = [self.row("g1"), self.row("g2", start=1900000600),
                self.row("g3", start=1900001200)]
        rows[1]["blue_team"], rows[1]["red_team"] = "Beta", "Alpha"
        got = candidate.score_rows(self.registration(), rows, bootstrap=100, now=2000000001)
        score = next(iter(got["by_incumbent_version"].values()))
        self.assertEqual(score["series_date_clusters"], 1)
        self.assertIsNone(score["ci95"])
        self.assertFalse(score["passed"])

    def test_hash_mismatch_failures_missing_unresolved_and_versions_are_separate(self):
        base = self.row("g1")
        second = dict(self.row("g2"), stack_sha256="v9", code_revision="code2")
        mismatch = dict(self.row("g3"), input_sha256="different")
        failure = dict(self.row("g4"), candidate_p=None, status="failed")
        missing = dict(self.row("g5"), candidate_p=None, status=None)
        unresolved = dict(self.row("g6"), blue_win=None)
        got = candidate.score_rows(self.registration(), [base, second, mismatch, failure, missing, unresolved],
                                   bootstrap=100, now=1900000200)
        self.assertEqual(got["resolved_paired_rows"], 2)
        self.assertEqual(len(got["by_incumbent_version"]), 2)
        self.assertEqual((got["failed_rows"], got["missing_rows"]), (1, 1))
        self.assertFalse(got["endpoint_complete"])
        self.assertTrue(all(not c["passed"] for c in got["by_incumbent_version"].values()))

    def test_direct_evidence_binds_frozen_plan_and_exact_incumbent(self):
        registration = self.registration()
        plan = {"provenance": {"incumbent_sha256": "incumbent", "incumbent_stack_sha256": "v8"},
                "inference_source": {"wpgam.py": "source"}, "rule": registration["rule"]}
        registration["frozen_plan"] = plan
        registration["plan_sha256"] = candidate.digest(plan)
        rows = [dict(self.row("g1"), model_sha256="incumbent"),
                dict(self.row("g2", start=1900086400), model_sha256="incumbent")]
        scored = candidate.score_rows(registration, rows, bootstrap=100, now=2000000001)
        evidence = next(iter(scored["by_incumbent_version"].values()))["direct_evidence"]
        self.assertEqual(evidence["frozen_plan"], plan)
        self.assertEqual(evidence["plan_sha256"], candidate.digest(plan))
        self.assertTrue(evidence["game_balanced"])
        self.assertEqual(evidence["clusters"], 2)
        rows[0]["model_sha256"] = "different"
        rows[1]["model_sha256"] = "different"
        scored = candidate.score_rows(registration, rows, bootstrap=0, now=2000000001)
        self.assertNotIn("direct_evidence", next(iter(scored["by_incumbent_version"].values())))


class CandidateDatabaseTests(unittest.TestCase):
    """Real PostgreSQL constraints, isolated in a disposable test schema."""
    @classmethod
    def setUpClass(cls):
        try:
            cls.conn = psycopg.connect(config.PG_DSN, row_factory=dict_row, autocommit=True, connect_timeout=2)
        except psycopg.OperationalError as exc:
            raise unittest.SkipTest("PostgreSQL unavailable for isolated candidate integration tests: %s" % exc)
        cls.schema = "test_candidate_" + uuid.uuid4().hex
        cls.conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(cls.schema)))
        cls.conn.execute("""CREATE TABLE shadow_predictions (
            protocol_id text, game_id text, game_start_ts bigint, minute int,
            captured_at timestamptz, feed_ts bigint, state jsonb,
            PRIMARY KEY (protocol_id,game_id,game_start_ts,minute))""")
        cls.conn.execute("CREATE TABLE shadow_outcomes (game_id text, game_start_ts bigint)")
        for statement in candidate.SCHEMA:
            cls.conn.execute(statement)

    @classmethod
    def tearDownClass(cls):
        cls.conn.execute("SET search_path TO public")
        cls.conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))
        cls.conn.close()

    def fixture(self, *, captured_age=0, game_age=60, known_outcome=False):
        now = time.time()
        cid = "test-" + uuid.uuid4().hex
        game = uuid.uuid4().hex
        reg = self.conn.execute("""INSERT INTO shadow_candidates
            (candidate_id, registered_at,label,artifact_path,artifact_sha256,model_kind,input_contract,
             input_contract_sha256,inference_source,inference_source_sha256,source_revision,
             training_cutoff,evaluation_end,rule,plan_sha256,artifact_meta)
            VALUES (%s,to_timestamp(%s),'test','test.npz','artifact','test','{}','contract','{}','source','revision',
                    to_timestamp(%s),to_timestamp(%s),'{}','plan','{}') RETURNING *""",
            (cid, now-300, now-1000, now+1000)).fetchone()
        state = {"ts": int(now-game_age+60), "gold_blue": 10000, "p_blue": .5}
        frame = {"protocol_id": "test", "game_id": game, "game_start_ts": int(now-game_age),
                 "minute": 1, "captured_at": now-captured_age, "state": state}
        self.conn.execute("""INSERT INTO shadow_predictions VALUES
            (%s,%s,%s,1,to_timestamp(%s),%s,%s)""",
            ("test", game, frame["game_start_ts"], now-captured_age, state["ts"], Json(state)))
        if known_outcome:
            self.conn.execute("INSERT INTO shadow_outcomes VALUES (%s,%s)", (game, frame["game_start_ts"]))
        return dict(reg), frame

    def test_forecasts_and_registration_are_immutable_and_duplicate_idempotent(self):
        reg, frame = self.fixture()
        candidate._write_prediction(self.conn, reg, frame, .6)
        candidate._write_prediction(self.conn, reg, frame, .9)
        saved = self.conn.execute("SELECT * FROM shadow_candidate_predictions WHERE candidate_id=%s",
                                  (reg["candidate_id"],)).fetchone()
        self.assertEqual(saved["candidate_p"], .6)
        self.assertEqual(saved["input_state"], frame["state"])
        self.assertEqual(saved["input_sha256"], candidate.digest(frame["state"]))
        for command in ("UPDATE shadow_candidate_predictions SET candidate_p=.9 WHERE candidate_id=%s",
                        "DELETE FROM shadow_candidate_predictions WHERE candidate_id=%s",
                        "UPDATE shadow_candidates SET artifact_sha256='changed' WHERE candidate_id=%s",
                        "DELETE FROM shadow_candidates WHERE candidate_id=%s"):
            with self.assertRaisesRegex(psycopg.errors.RaiseException, "append-only"):
                self.conn.execute(command, (reg["candidate_id"],))

    def test_database_rejects_backfill_pre_registration_games_and_known_outcomes(self):
        for kwargs in ({"captured_age": 20}, {"game_age": 600}, {"known_outcome": True}):
            with self.subTest(kwargs=kwargs):
                reg, frame = self.fixture(**kwargs)
                with self.assertRaises(psycopg.errors.RaiseException):
                    candidate._write_prediction(self.conn, reg, frame, .6)

    def test_database_rejects_changed_inputs_and_provenance(self):
        reg, frame = self.fixture()
        changed = copy.deepcopy(frame)
        changed["state"]["gold_blue"] += 1
        with self.assertRaisesRegex(psycopg.errors.RaiseException, "provenance mismatch"):
            candidate._write_prediction(self.conn, reg, changed, .6)
        for key in ("artifact_sha256", "inference_source_sha256"):
            with self.assertRaisesRegex(psycopg.errors.RaiseException, "provenance mismatch"):
                candidate._write_prediction(self.conn, dict(reg, **{key: "changed"}), frame, .6)

    def test_registration_freezes_bytes_cutoff_contract_and_plan(self):
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = now-dt.timedelta(days=1)
        model = {"kind": "test", "state": {"feature_names": ["gold_k"]},
                 "meta": {"training_cutoff": cutoff.isoformat(), "input_contract": {"version": "causal"},
                          "dataset_sha256": "corrected-dataset"}}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "candidate.npz")
            with open(path, "wb") as stream:
                stream.write(b"frozen model bytes")
            with mock.patch.object(candidate, "FREEZE_DIR", os.path.join(directory, "frozen")), \
                    mock.patch.object(candidate, "ensure_schema"), \
                    mock.patch.object(candidate.wpgam, "load_model", return_value=model):
                kwargs = dict(training_cutoff=cutoff, evaluation_end=now+dt.timedelta(days=30),
                              min_gain_brier=.001, precision_target=.002,
                              sample_size_basis={"pilot": "corrected development"},
                              incumbent_versions={"model_sha256": "incumbent", "stack_sha256": "incumbent-stack"})
                first = candidate.register(self.conn, path, **kwargs)
                second = candidate.register(self.conn, path, **kwargs)
                self.assertEqual(first["candidate_id"], second["candidate_id"])
                self.assertEqual(first["registered_at"], second["registered_at"])
                self.assertEqual(candidate.sha256(first["artifact_path"]), first["artifact_sha256"])
                self.assertEqual(first["input_contract_sha256"], candidate.digest(first["input_contract"]))
                self.assertEqual(first["plan_sha256"], candidate.digest(first["frozen_plan"]))
                self.assertIn("operational", first["rule"]["minimum_games_role"])
                with self.assertRaisesRegex(ValueError, "training_cutoff"):
                    candidate.register(self.conn, path, **dict(kwargs, training_cutoff=cutoff-dt.timedelta(days=1)))


if __name__ == "__main__":
    unittest.main()
