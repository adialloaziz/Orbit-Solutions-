import unittest
import os
import importlib
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.linalg import expm

from scripts.utility_2 import orbit

VFP_PYTHON_DIR = Path(__file__).resolve().parents[1] / "VFP_bind" / "python"
sys.path.insert(0, str(VFP_PYTHON_DIR))
vfp_module: Any = None
try:
    vfp_module = importlib.import_module("module_vfp")
except ImportError:
    pass


class FlowMonodromyTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[0.0, 1.0], [-2.0, -0.5]])
        self.rate = 1.3
        self.flow = lambda y, T, rate: expm(rate * T * self.matrix) @ y
        self.y = np.array([0.4, -0.7])
        self.period = 0.8
        self.orbit = orbit(self.flow, self.y, {"rate": self.rate})

    def test_monodromy_action_matches_linear_flow(self):
        vector = np.array([1.2, 0.3])
        actual = self.orbit.monodromy_mult(
            self.y, self.period, vector, epsilon=1e-7
        )
        expected = expm(self.rate * self.period * self.matrix) @ vector

        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

    def test_integrated_monodromy_matches_linear_flow(self):
        actual_state, actual_matrix = self.orbit.integ_monodromy(
            self.y, np.eye(self.y.size), self.period
        )
        expected_matrix = expm(self.rate * self.period * self.matrix)

        np.testing.assert_allclose(actual_state, expected_matrix @ self.y)
        np.testing.assert_allclose(
            actual_matrix, expected_matrix, rtol=2e-6, atol=2e-7
        )


@unittest.skipIf(vfp_module is None, "Build the VFP Python extension to run this test")
class VFPFlowMonodromyTests(unittest.TestCase):
    def test_monodromy_action_matches_direct_vfp_flow_difference(self):
        grid_args = {
            "nt": 2,
            "ntint": 2,
            "n_q": 8,
            "n_p": 8,
            "Lq": 20.0,
            "Lp": 20.0,
            "grand_L": 40.0,
        }

        def vfp_flow(y, T, *, nt, ntint, n_q, n_p, Lq, Lp, grand_L):
            result = vfp_module.integ_vfp(
                np.asarray(y, dtype=np.float64).copy(),
                0.0,
                T,
                nt,
                ntint,
                n_q,
                n_p,
                Lq,
                Lp,
                grand_L,
            )
            return result.y

        previous_directory = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as working_directory:
                os.chdir(working_directory)
                vfp_module.init_WF(
                    grid_args["n_q"], grid_args["Lq"], grid_args["grand_L"]
                )
                initial_state = vfp_module.init_vfp(
                    grid_args["n_q"],
                    grid_args["n_p"],
                    grid_args["Lq"],
                    grid_args["Lp"],
                    1.2,
                    1.2,
                )
                direction = np.random.default_rng(0).standard_normal(initial_state.size)
                direction /= np.linalg.norm(direction)
                period = 1e-3
                epsilon = 1e-6
                initial_copy = initial_state.copy()
                orbit_finder = orbit(vfp_flow, initial_state, grid_args)

                actual = orbit_finder.monodromy_mult(
                    initial_state, period, direction, epsilon=epsilon
                )
                base_flow = vfp_flow(initial_state, period, **grid_args)
                perturbed_flow = vfp_flow(
                    initial_state + epsilon * direction, period, **grid_args
                )
                expected = (perturbed_flow - base_flow) / epsilon

                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(actual.shape, initial_state.shape)
                self.assertTrue(np.all(np.isfinite(actual)))
                np.testing.assert_array_equal(initial_state, initial_copy)
        finally:
            os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()