import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causalshift import ExperimentConfig, generate_stream, two_proportion_pvalue


class TestCausalShift(unittest.TestCase):
    def test_generator_shape_and_shift(self):
        config = ExperimentConfig(n_samples=600, shift_point=300, seed=7)
        df = generate_stream(config)
        self.assertEqual(len(df), 600)
        self.assertEqual(int(df["true_shift"].sum()), 300)
        self.assertEqual(df["y"].isin([0, 1]).all(), True)

    def test_pvalue_same_rates_is_not_significant(self):
        a = np.array([0, 1] * 50)
        b = np.array([0, 1] * 50)
        self.assertGreaterEqual(two_proportion_pvalue(a, b), 0.99)

    def test_config_rejects_too_small_stream(self):
        with self.assertRaises(ValueError):
            generate_stream(ExperimentConfig(n_samples=100, shift_point=50))

if __name__ == "__main__":
    unittest.main()
