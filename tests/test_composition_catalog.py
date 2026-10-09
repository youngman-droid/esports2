import json
import tempfile
import unittest
from pathlib import Path

from scripts import composition_catalog as catalog


class Response:
    def __init__(self, version="16.9.1", modified="Tue, 28 Apr 2026 20:28:26 GMT"):
        self.raw = json.dumps(dict(version=version, data=dict(Aatrox=dict(name="Aatrox", tags=["Fighter"], stats={})))).encode()
        self.headers = {"Last-Modified": modified} if modified is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.raw


class CatalogTests(unittest.TestCase):
    def test_exact_release_raw_hash_timing_and_cache_replay(self):
        calls = []

        def opener(url, timeout):
            calls.append(url)
            self.assertEqual(timeout, 30)
            return Response()

        with tempfile.TemporaryDirectory() as directory:
            result = catalog.fetch("16.9.1", directory, opener=opener)
            self.assertEqual(result["timing_source"], "http_last_modified")
            self.assertEqual(result["version"], "16.9.1")
            self.assertEqual(catalog.fetch("16.9.1", directory, opener=opener), result)
            self.assertEqual(len(calls), 1)
            (Path(directory) / "raw" / "16.9.1.json").write_text("tampered")
            with self.assertRaises(ValueError):
                catalog.fetch("16.9.1", directory, opener=opener)

    def test_wrong_version_missing_source_clock_and_mutable_patch_urls_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            for version, response in (("16.9.1", Response("16.10.1")),
                                       ("16.9.1", Response(modified=None)), ("16.9", Response())):
                with self.assertRaises(ValueError):
                    catalog.fetch(version, directory, opener=lambda *args, **kwargs: response)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
