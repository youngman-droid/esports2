import tempfile
import unittest
from pathlib import Path

import numpy as np

from research import wpx_recover_v8 as recovery


class ExactRecoveryGuardsTests(unittest.TestCase):
    def test_rejects_modified_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.npz"
            artifact.write_bytes(b"verified model contents")
            expected = recovery.sha(artifact)
            artifact.write_bytes(b"changed model contents")
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                recovery.require_hash(artifact, expected)

    def test_rejects_prediction_drift_and_nonfinite_values(self):
        expected = np.array([.3, .6])
        for actual in ([.3, .6000001], [.3, np.nan], [.3]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                recovery.validate_predictions(actual, expected)


if __name__ == "__main__":
    unittest.main()
