import unittest
import os
import importlib
from pathlib import Path
import sys
import tempfile
from inspect import Parameter, signature
from typing import Any

import numpy as np
from scipy.linalg import expm

from utility_2 import orbit
from model import VFPModel

VFP_PYTHON_DIR = Path(__file__).resolve().parents[1] / "VFP_bind" / "python"
sys.path.insert(0, str(VFP_PYTHON_DIR))
vfp_module: Any = None
try:
    vfp_module = importlib.import_module("module_vfp")
except ImportError:
    pass


def mass_weights(n_q, n_p, Lq, Lp):
    q_weights = np.ones(n_q)
    p_weights = np.ones(n_p)
    q_weights[[0, -1]] = 0.5
    p_weights[[0, -1]] = 0.5
    return (Lq / n_q) * (Lp / n_p) * np.kron(q_weights, p_weights)


class FlowMonodromyTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[0.0, 1.0], [-2.0, -0.5]])
        self.rate = 1.3

        class LinearModel:
            def __init__(self, matrix, rate):
                self.matrix = matrix
                self.solver_args = {"rate": rate}

            def flow(self, y, T, rate):
                return expm(rate * T * self.matrix) @ y

            def flow_direction(self, y, delta=1e-6):
                return (self.flow(y, delta, **self.solver_args) - y) / delta
            def flow_time_derivative(self, y, T, delta=1e-6):
                return (self.flow(y,  T+delta, **self.solver_args) - self.flow(y, T, **self.solver_args)) / delta

            def flow_parameter_derivative(self, y, T, parameter, delta_param=1e-6):
                try:
                    parameter_info = signature(self.flow).parameters.get(parameter)
                except (TypeError, ValueError) as error:
                    raise TypeError("The flow signature must be inspectable") from error

                if parameter_info is None or parameter_info.kind is Parameter.POSITIONAL_ONLY:
                    raise ValueError(
                        f"{parameter!r} is not a keyword parameter in the flow signature"
                    )
                if parameter not in self.solver_args:
                    raise ValueError(f"{parameter!r} is not present in solver_args")

                args_plus = dict(self.solver_args)
                args_plus[parameter] += delta_param
                flow_plus = self.flow(y, T, **args_plus)
                flow_base = self.flow(y, T, **self.solver_args)
                return (flow_plus - flow_base) / delta_param

        self.model = LinearModel(self.matrix, self.rate)
        self.y = np.array([0.4, -0.7])
        self.period = 0.8
        self.orbit = orbit(self.model, self.y, {"rate": self.rate})

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

    def test_flow_parameter_derivative_matches_linear_flow(self):
        actual = self.orbit.flow_parameter_derivative(
            self.y, self.period, "rate", delta_param=1e-6
        )
        exact_flow = expm(self.rate * self.period * self.matrix) @ self.y
        expected = self.period * self.matrix @ exact_flow

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


    def test_newton_orbit_finds_known_period(self):
        class PeriodicModel:
            def flow(self, state, period):
                angle, radius, stable = state
                return np.array([
                    (angle + period + np.pi) % (2 * np.pi) - np.pi,
                    1 + (radius - 1) * np.exp(-period),
                    stable * np.exp(-period),
                ])

            def flow_direction(self, state, delta=1e-6):
                return (self.flow(state, delta) - state) / delta

            def flow_time_derivative(self, state, period, delta=1e-6):
                return (self.flow(state, period + delta) - self.flow(state, period)) / delta

            def flow_parameter_derivative(self, state, period, parameter, delta_param=1e-6):
                raise ValueError(f"This flow has no parameter {parameter!r}")

        initial_state = np.array([1.5, 1.2, 1.4])
        orbit_finder = orbit(PeriodicModel(), initial_state)
        result = orbit_finder.Newton_orbit(
            initial_state, 2.0, Max_iter=10, epsilon=1e-8
        )
        iteration, periods, states, _, _, _, converged, _ = result

        self.assertEqual(converged, 1)
        period = periods[iteration]
        state = states[iteration]
        self.assertAlmostEqual(period, 2 * np.pi, places=6)
        np.testing.assert_allclose(orbit_finder.flow(state, period), state, atol=1e-8)


@unittest.skipIf(vfp_module is None, "Build the VFP Python extension to run this test")
class VFPFlowMonodromyTests(unittest.TestCase):
    def test_monodromy_action_matches_direct_vfp_flow_difference(self):
        solv_args = {
            "I": 8.044e-5*2.,
            "alpha_unfold": 1e-6,
            "H": mass_weights(8, 8, 20.0, 20.0),
            "nt": 2,
            "ntint": 2,
            "n_q": 8,
            "n_p": 8,
            "Lq": 20.0,
            "Lp": 20.0,
            "grand_L": 40.0,
            
        }
        rms_gauss_q = 1.2
        rms_gauss_p = 1.2
        
        def vfp_flow(y, T, I, alpha_unfold, *, H, nt, ntint, n_q, n_p, Lq, Lp, grand_L):
            result = vfp_module.integ_vfp(
                np.asarray(y, dtype=np.float64).copy(),
                H,
                0.0,
                T,
                I,
                alpha_unfold,
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
                    solv_args["n_q"], solv_args["Lq"], solv_args["grand_L"]
                )

                model = VFPModel(solver_args=solv_args)
                y0 = vfp_module.init_vfp(
                    solv_args["n_q"],
                    solv_args["n_p"],
                    solv_args["Lq"],
                    solv_args["Lp"],
                    rms_gauss_q,
                    rms_gauss_p,
                ) #Initial state
                direction = np.random.default_rng(0).standard_normal(y0.size)
                direction /= np.linalg.norm(direction)

                
                T = 1e-3
                epsilon = 1e-6
                y0_copy = y0.copy()
                orbit_finder = orbit(model, y0, solv_args)

                actual = orbit_finder.monodromy_mult(
                    y0, T, direction, epsilon=epsilon
                )
                base_flow = vfp_flow(y0, T,**solv_args)
                perturbed_flow = vfp_flow(
                    y0 + epsilon * direction, T, **solv_args
                )
                expected = (perturbed_flow - base_flow) / epsilon

                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(actual.shape, y0.shape)
                self.assertTrue(np.all(np.isfinite(actual)))
                np.testing.assert_array_equal(y0, y0_copy)
        finally:
            os.chdir(previous_directory)

    def test_newton_orbit_runs_with_vfp_solver(self):
        solv_args = {
            "I": 8.044e-5*2.,
            "alpha_unfold": 1e-6,
            "H": mass_weights(10, 10, 20.0, 20.0),
            "nt": 100,
            "ntint": 10,
            "n_q": 10,
            "n_p": 10,
            "Lq": 20.0,
            "Lp": 20.0,
            "grand_L": 40.0,
        }

        previous_directory = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as working_directory:
                os.chdir(working_directory)
                vfp_module.init_WF(
                    solv_args["n_q"], solv_args["Lq"], solv_args["grand_L"]
                )
                model = VFPModel(solver_args=solv_args)

                y0 = vfp_module.init_vfp(
                    solv_args["n_q"], solv_args["n_p"], solv_args["Lq"],
                    solv_args["Lp"], 1.2, 1.2,
                )
                y0_copy = y0.copy()
                orbit_finder = orbit(model, y0, solv_args)
                T0 = 3
                result = orbit_finder.Newton_orbit(
                    y0, T0, Max_iter=10, epsilon=1e-8
                )

                iteration, periods, states, _, _, _, converged, _ = result
                # self.assertEqual(iteration, 0)
                self.assertEqual(periods.shape, (10,))
                self.assertEqual(states.shape, (10, y0.size))
                self.assertIn(converged, (-1, 0, 1))
                self.assertTrue(np.all(np.isfinite(periods)))
                self.assertTrue(np.all(np.isfinite(states)))
                np.testing.assert_array_equal(y0, y0_copy)
        finally:
            os.chdir(previous_directory)

    def test_newton_picard_sub_proj_runs_with_vfp_solver(self):
        solv_args = {
            "I": 8.044e-5*2.,
            "alpha_unfold": 1e-6,
            "H": mass_weights(40, 40, 20.0, 20.0),
            "nt": 20,
            "ntint": 2,
            "n_q": 40,
            "n_p": 40,
            "Lq": 20.0,
            "Lp": 20.0,
            "grand_L": 40.0,
        }


        previous_directory = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as working_directory:
                os.chdir(working_directory)
                vfp_module.init_WF(
                    solv_args["n_q"], solv_args["Lq"], solv_args["grand_L"]
                )
                y0 = vfp_module.init_vfp(
                    solv_args["n_q"], solv_args["n_p"], solv_args["Lq"],
                    solv_args["Lp"], 1.2, 1.2,
                )
                model = VFPModel(solver_args=solv_args)

                orbit_finder = orbit(model, y0, solv_args)
                flow_direction = orbit_finder.flow_direction(y0)
                flow_direction /= np.linalg.norm(flow_direction)
                p0, pe = 5, 4
                basis_size = p0 + pe
                coordinate_indices = np.argsort(np.abs(flow_direction))[:basis_size - 1]
                transverse = np.zeros((y0.size, basis_size - 1))
                transverse[coordinate_indices, np.arange(basis_size - 1)] = 1.0
                initial_basis, _ = np.linalg.qr(
                    np.column_stack((flow_direction, transverse)), mode="reduced"
                )
                self.assertEqual(initial_basis.shape, (y0.size, basis_size))
                np.testing.assert_allclose(
                    initial_basis.T @ initial_basis,
                    np.eye(basis_size),
                    atol=1e-12,
                )
                y0_copy = y0.copy()
                T0 = 3
                result = orbit_finder.Newton_Picard_sub_proj(
                    y0,
                    T0,
                    Max_iter=10,
                    epsilon=1e-8,
                    subsp_iter=5,
                    Ve_0=initial_basis,
                    p0=p0,
                    pe=pe,
                    rho=0.5,
                    l=2,
                )
                iteration, periods, states, residuals, abs_errors, rel_errors, converged, masses = result
                self.assertEqual(periods.shape, (10,))
                self.assertEqual(states.shape, (10, y0.size))
                self.assertEqual(residuals.shape, (10,))
                self.assertEqual(abs_errors.shape, (10,))
                self.assertIn(converged, (-1, 0, 1))
                self.assertTrue(np.all(np.isfinite(periods)))
                self.assertTrue(np.all(np.isfinite(states)))
                np.testing.assert_array_equal(y0, y0_copy)
        finally:
            os.chdir(previous_directory)

    def test_np_mass_conserv_runs_with_vfp_solver(self):
        n_q = 40
        n_p = 40
        Lq = 20.0
        Lp = 20.0
        alpha_unfold = 1e-6
        solv_args = {
            "I": 8.044e-5 * 2.0,
            "alpha_unfold": alpha_unfold,
            "H": mass_weights(n_q, n_p, Lq, Lp),
            "nt": 20,
            "ntint": 2,
            "n_q": n_q,
            "n_p": n_p,
            "Lq": Lq,
            "Lp": Lp,
            "grand_L": 40.0,
        }

        previous_directory = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as working_directory:
                os.chdir(working_directory)
                vfp_module.init_WF(n_q, Lq, solv_args["grand_L"])
                y0 = vfp_module.init_vfp(n_q, n_p, Lq, Lp, 1.2, 1.2)
                model = VFPModel(solver_args=solv_args)
                orbit_finder = orbit(model, y0, solv_args)

                flow_direction = orbit_finder.flow_direction(y0)
                flow_direction /= np.linalg.norm(flow_direction)
                p0, pe = 5, 4
                basis_size = p0 + pe
                coordinate_indices = np.argsort(np.abs(flow_direction))[:basis_size - 1]
                transverse = np.zeros((y0.size, basis_size - 1))
                transverse[coordinate_indices, np.arange(basis_size - 1)] = 1.0
                initial_basis, _ = np.linalg.qr(
                    np.column_stack((flow_direction, transverse)), mode="reduced"
                )
                y0_copy = y0.copy()

                result = orbit_finder.NP_mass_conserv(
                    model=model,
                    y_0=y0,
                    T_0=3.0,
                    I_0=solv_args["I"],
                    alpha_unfold=alpha_unfold,
                    Max_iter=10,
                    epsilon=1e-8,
                    H=solv_args["H"],
                    subsp_iter=1,
                    Ve_0=initial_basis,
                    p0=p0,
                    pe=pe,
                    rho=0.5,
                    l=2,
                )

                iteration, periods, states, residuals, abs_errors, rel_errors, converged, masses = result
            
                self.assertEqual(periods.shape, (10,))
                self.assertEqual(states.shape, (10, y0.size))
                self.assertEqual(residuals.shape, (10,))
                self.assertEqual(abs_errors.shape, (10,))
                self.assertEqual(rel_errors.shape, (10,))
                self.assertIn(converged, (-1, 0, 1))
                self.assertTrue(np.all(np.isfinite(periods)))
                self.assertTrue(np.all(np.isfinite(states)))
                np.testing.assert_array_equal(y0, y0_copy)
        finally:
            os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()