import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.risk_engine import FEATURE_ORDER, build_feature_vector, score_with_fallback


class RiskEngineTests(unittest.TestCase):
    def test_feature_vector_uses_expected_defaults(self):
        features = {
            "vegetation_index": 0.2,
            "flood_proximity_score": 3,
            "crime_risk_index": 2.5,
        }
        vector = build_feature_vector(features)
        self.assertEqual(len(vector), len(FEATURE_ORDER))
        self.assertEqual(vector[0], 0.2)
        self.assertEqual(vector[1], 3.0)
        self.assertEqual(vector[8], 2.5)
        self.assertEqual(vector[9], 0.0)
        self.assertEqual(vector[10], 0.0)

    def test_score_with_fallback_returns_clamped_score(self):
        score = score_with_fallback(None, {"vegetation_index": 0.2, "crime_risk_index": 3.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


if __name__ == '__main__':
    unittest.main()
