import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from lol_ticker import __main__ as cli


class CorrectedInputCommandTests(unittest.TestCase):
    def rejected_before_database(self, args):
        with mock.patch("sys.argv", ["lol_ticker", "wpx"] + args), \
                mock.patch.object(cli.db, "connect") as connect, \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as stopped:
            cli.main()
        self.assertEqual(stopped.exception.code, 2)
        connect.assert_not_called()

    def test_ambiguous_full_pipeline_and_implicit_build_fail_before_database(self):
        self.rejected_before_database(["all"])
        self.rejected_before_database(["build"])
        self.rejected_before_database(["build", "--out", "new.npz"])
        self.rejected_before_database(["build", "--before", "2026-09-03"])

    def test_bad_cutoff_and_build_options_on_other_steps_fail_before_database(self):
        self.rejected_before_database(["build", "--out", "new.npz", "--before", "2026-02-31"])
        self.rejected_before_database(["build", "--out", "new.npz", "--before", "2026-9-3"])
        self.rejected_before_database(["fit", "--out", "new.npz"])

    def test_explicit_build_hands_off_path_and_cutoff_without_fitting(self):
        args = ["lol_ticker", "wpx", "build", "--out", "new.npz", "--before", "2026-09-03"]
        with mock.patch("sys.argv", args), mock.patch.object(cli.db, "connect") as connect, \
                mock.patch("lol_ticker.wpx.build", return_value="new.npz") as build, \
                mock.patch("lol_ticker.wpx.fit_full") as fit, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main()
        build.assert_called_once_with(connect.return_value, output_path="new.npz", before="2026-09-03")
        fit.assert_not_called()
        self.assertIn("Default fit/eval inputs are unchanged", output.getvalue())

    def test_postdraft_requires_explicit_output_and_cutoff(self):
        self.rejected_before_database(["postdraft-build"])
        self.rejected_before_database(["postdraft-build", "--out", "new.npz"])
        self.rejected_before_database(["postdraft-build", "--out", "new.npz", "--before", "2026-02-31"])

    def test_postdraft_build_dispatches_without_gameplay_builder_or_fit(self):
        args = ["lol_ticker", "wpx", "postdraft-build", "--out", "new.npz", "--before", "2026-09-03"]
        with mock.patch("sys.argv", args), mock.patch.object(cli.db, "connect") as bootstrap, \
                mock.patch.object(cli.psycopg, "connect") as connect, \
                mock.patch("lol_ticker.wppostdraft.build", return_value="new.npz") as build, \
                mock.patch("lol_ticker.wpx.build") as gameplay, \
                mock.patch("lol_ticker.wpx.fit_full") as fit, \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        build.assert_called_once_with(connect.return_value, output_path="new.npz", before="2026-09-03")
        bootstrap.assert_not_called()
        self.assertEqual(connect.call_args.kwargs["options"], "-c default_transaction_read_only=on")
        connect.return_value.execute.assert_called_once_with("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        gameplay.assert_not_called()
        fit.assert_not_called()


class ScheduledUpdateTests(unittest.TestCase):
    def test_normal_refresh_keeps_ingestion_and_shadow_scoring_without_model_refresh(self):
        # Execute the real orchestration with a command recorder in place of
        # Python; no network request, DB connection or model artifact is used.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "data" / "oe").mkdir(parents=True)
            (root / "bin").mkdir()
            source = Path(__file__).resolve().parents[1] / "scripts" / "update.sh"
            shutil.copyfile(source, root / "scripts" / "update.sh")
            fake_python = root / "bin" / "python3"
            fake_python.write_text(
                '#!/bin/sh\n'
                'printf "%s\\n" "$*" >> "$UPDATE_TEST_CALLS"\n'
                'if [ "$1" = "scripts/drive_download.py" ]; then : > "$3"; fi\n'
            )
            fake_python.chmod(0o755)
            calls_path = root / "calls.txt"
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ.get("PATH", ""),
                       UPDATE_TEST_CALLS=str(calls_path))
            run = subprocess.run(["/bin/sh", str(root / "scripts" / "update.sh"), "--no-record"],
                                 cwd=root, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            calls = calls_path.read_text().splitlines()
            for command in ("discover", "backfill", "draftload", "align", "wpa", "draftfree", "wpx prep", "shadow score"):
                self.assertIn("-m lol_ticker " + command, calls)
            self.assertTrue(any(c.startswith("-m lol_ticker golgg ") for c in calls))
            self.assertIn("scripts/feed_backfill.py --workers 6", calls)
            self.assertFalse(any("wpx build" in c or "wpx fit" in c or "wpx_eval.py" in c for c in calls))
            self.assertIn("gated pending corrected-input candidate evidence", run.stdout)
            self.assertIn("shadow outcomes resolved and scored", run.stdout)


if __name__ == "__main__":
    unittest.main()
