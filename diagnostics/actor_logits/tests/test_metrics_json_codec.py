"""Regression: real evaluator metrics may contain NumPy arrays.

CPU only; no policy weight, simulator, GPU, extra forward or episode.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from diagnostics.actor_logits.storage import json_value, read_json, write_json


class MetricsNumpyJsonRegression(unittest.TestCase):
    def test_nested_metrics_ndarray_roundtrip(self):
        metrics = {
            "official": {
                "success": np.float32(1.0),
                "eps_len": np.int64(27),
                "cost": 3,
                "visit_vector": np.array([0, 1, 2], dtype=np.int64),
                "camera_visibility": np.array([[0.125, 0.5], [0.75, 1.0]], dtype=np.float32),
                "scalar_array": np.asarray(2.5),
            },
            "summary": {"episode_order": 0, "success": True},
        }
        with TemporaryDirectory() as d:
            p = Path(d) / "metrics.json"
            write_json(p, metrics)
            actual = read_json(p)
            self.assertEqual(actual["official"]["visit_vector"], [0, 1, 2])
            self.assertEqual(actual["official"]["camera_visibility"],
                             [[0.125, 0.5], [0.75, 1.0]])
            self.assertEqual(actual["official"]["scalar_array"], 2.5)
            self.assertEqual(actual["official"]["eps_len"], 27)
            self.assertEqual(actual["summary"]["episode_order"], 0)
            self.assertTrue(p.read_text().endswith("\\n"))

    def test_ndarray_empty_and_nested(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "metrics.json"
            write_json(p, {"empty": np.array([], dtype=np.float32),
                           "labels": np.array(["walk", "end"]),
                           "count": np.int64(0)})
            self.assertEqual(read_json(p), {"empty": [], "labels": ["walk", "end"], "count": 0})

    def test_nonfinite_remains_explicit_failure(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_json(Path(d) / "metrics.json",
                           {"official": {"cost": np.array([np.nan], dtype=np.float32)}})

    def test_unsupported_dtype_remains_failure(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(TypeError):
                write_json(Path(d) / "metrics.json",
                           {"official": {"complex": np.array([1 + 2j])}})


if __name__ == "__main__":
    unittest.main()
