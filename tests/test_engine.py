import unittest

import numpy as np
import sympy as sp

from eml import analyze, regress
from eml.engine import C, ONE, X, ZERO, evaluate_tree, tree_to_sympy


def _eval_sympy(expr, x_data, feature_names=None):
    if expr is None:
        raise AssertionError("Expression should not be None")

    x_arr = np.asarray(x_data, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
        names = list(feature_names or ["x"])
    elif x_arr.ndim == 2:
        names = list(feature_names or [f"x{i}" for i in range(x_arr.shape[1])])
    else:
        raise AssertionError("x_data must be 1D or 2D")

    if not expr.free_symbols:
        return np.full(x_arr.shape[0], float(expr), dtype=np.float64)

    free_symbols = {sym.name: sym for sym in expr.free_symbols}
    ordered_names = [name for name in names if name in free_symbols]
    ordered_symbols = [free_symbols[name] for name in ordered_names]
    fn = sp.lambdify(ordered_symbols, expr, "numpy")
    values = [x_arr[:, names.index(name)] for name in ordered_names]
    evaluated = fn(*values)
    if np.isscalar(evaluated):
        return np.full(x_arr.shape[0], float(evaluated), dtype=np.float64)
    return np.asarray(evaluated, dtype=np.float64)


class TreeToSympyTests(unittest.TestCase):
    def test_negative_constant_uses_abs_not_complex_log(self):
        x_data = np.linspace(0.1, 2.0, 32)
        x_symbol = sp.Symbol("x", positive=True, real=True)
        expr = tree_to_sympy([X, C], constants=[-2.0], x_symbol=x_symbol)

        self.assertIsNotNone(expr)
        self.assertNotIn("I", str(expr))
        self.assertNotIn("eps", str(expr))
        np.testing.assert_allclose(
            _eval_sympy(expr, x_data),
            evaluate_tree([X, C], x_data, [-2.0]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_zero_log_argument_returns_none(self):
        x_symbol = sp.Symbol("x", positive=True, real=True)
        self.assertIsNone(tree_to_sympy([X, ZERO], x_symbol=x_symbol))

    def test_symbolic_conversion_matches_numeric_tree_evaluation_in_1d_and_2d(self):
        x_data = np.linspace(0.1, 3.0, 64)
        x_matrix = np.column_stack([
            np.linspace(0.1, 2.0, 64),
            np.linspace(1.1, 3.1, 64),
        ])
        x_symbol = sp.Symbol("x", positive=True, real=True)
        feature_symbols = {
            "x0": sp.Symbol("x0", positive=True, real=True),
            "x1": sp.Symbol("x1", positive=True, real=True),
        }
        cases = [
            ([X, ONE], [], x_data, None, {"x": x_symbol}),
            ([ONE, X], [], x_data, None, {"x": x_symbol}),
            ([X, C], [-2.0], x_data, None, {"x": x_symbol}),
            (["x0", "x1"], [], x_matrix, ["x0", "x1"], feature_symbols),
            ([ONE, C, "x0", "x1"], [-2.0], x_matrix, ["x0", "x1"], feature_symbols),
        ]

        for leaf_types, constants, features, feature_names, symbols in cases:
            with self.subTest(leaf_types=leaf_types, constants=constants):
                expr = tree_to_sympy(leaf_types, constants=constants, feature_symbols=symbols)
                self.assertIsNotNone(expr)
                np.testing.assert_allclose(
                    _eval_sympy(expr, features, feature_names=feature_names),
                    evaluate_tree(leaf_types, features, constants, feature_names=feature_names),
                    rtol=1e-9,
                    atol=1e-9,
                )


class RegressionTests(unittest.TestCase):
    @staticmethod
    def _feature_grid(x0_values, x1_values):
        grid0, grid1 = np.meshgrid(x0_values, x1_values)
        return grid0.ravel(), grid1.ravel()

    def test_exact_regression_finds_standard_forms(self):
        x_data = np.linspace(0.1, 4.0, 200)
        targets = {
            "exp(x)": np.exp(x_data),
            "1/x": 1.0 / x_data,
            "ln(x)": np.log(x_data),
            "exp(-x)": np.exp(-x_data),
            "x + 1": x_data + 1.0,
            "x^2": x_data ** 2,
            "sqrt(x)": np.sqrt(x_data),
        }

        for name, y_data in targets.items():
            with self.subTest(name=name):
                result = regress(x_data, y_data, max_depth=3, verbose=False)
                self.assertLess(result["mse"], 1e-8)
                self.assertEqual(result["strategy"], "prepass")
                self.assertEqual(result["feature_names"], ["x"])
                self.assertEqual(result["used_features"], ["x"])
                for banned in ("eps", "I*pi", "zoo", "nan"):
                    self.assertNotIn(banned, result["expression"] or "")

    def test_multivariable_prepass_finds_common_two_feature_forms(self):
        cases = [
            ("x0 + x1", *self._feature_grid(np.linspace(0.1, 2.0, 12), np.linspace(0.5, 2.5, 10)),
             lambda x0, x1: x0 + x1),
            ("x0 * x1", *self._feature_grid(np.linspace(0.1, 1.5, 12), np.linspace(0.5, 2.0, 10)),
             lambda x0, x1: x0 * x1),
            ("x0 / x1", *self._feature_grid(np.linspace(0.2, 1.5, 12), np.linspace(1.0, 2.5, 10)),
             lambda x0, x1: x0 / x1),
            ("2*exp(x0 + x1)", *self._feature_grid(np.linspace(0.1, 0.8, 12), np.linspace(0.2, 0.9, 10)),
             lambda x0, x1: 2.0 * np.exp(x0 + x1)),
            ("x0**2 + x1", *self._feature_grid(np.linspace(0.1, 1.5, 12), np.linspace(0.5, 2.0, 10)),
             lambda x0, x1: x0 ** 2 + x1),
        ]

        for name, x0, x1, fn in cases:
            with self.subTest(name=name):
                x_matrix = np.column_stack([x0, x1])
                y_data = fn(x0, x1)
                result = regress(x_matrix, y_data, max_depth=2, verbose=False)
                self.assertLess(result["mse"], 1e-8)
                self.assertEqual(result["strategy"], "prepass")
                self.assertEqual(result["feature_names"], ["x0", "x1"])
                self.assertEqual(result["used_features"], ["x0", "x1"])
                self.assertIn("x0", result["expression"] or "")
                self.assertIn("x1", result["expression"] or "")

    def test_matrix_input_supports_custom_feature_names(self):
        length, time = self._feature_grid(np.linspace(0.2, 1.8, 10), np.linspace(1.0, 2.5, 8))
        x_matrix = np.column_stack([length, time])
        result = regress(
            x_matrix,
            length + time,
            max_depth=2,
            verbose=False,
            feature_names=["length", "time"],
        )

        self.assertLess(result["mse"], 1e-8)
        self.assertEqual(result["strategy"], "prepass")
        self.assertEqual(result["feature_names"], ["length", "time"])
        self.assertEqual(result["used_features"], ["length", "time"])
        self.assertIn("length", result["expression"] or "")
        self.assertIn("time", result["expression"] or "")

    def test_three_variable_prepass_finds_common_power_law_form(self):
        m1, m2, r = np.meshgrid(
            np.linspace(1.0, 4.0, 8),
            np.linspace(2.0, 5.0, 7),
            np.linspace(1.0, 3.0, 6),
        )
        x_matrix = np.column_stack([m1.ravel(), m2.ravel(), r.ravel()])
        y_data = (m1.ravel() * m2.ravel()) / (r.ravel() ** 2)

        result = regress(
            x_matrix,
            y_data,
            max_depth=3,
            verbose=False,
            feature_names=["m1", "m2", "r"],
        )

        self.assertLess(result["mse"], 1e-8)
        self.assertEqual(result["strategy"], "prepass")
        self.assertEqual(result["feature_names"], ["m1", "m2", "r"])
        self.assertEqual(result["used_features"], ["m1", "m2", "r"])
        self.assertIn("m1", result["expression"] or "")
        self.assertIn("m2", result["expression"] or "")
        self.assertIn("r", result["expression"] or "")

    def test_analyze_routes_common_families(self):
        x_data = np.linspace(0.2, 4.0, 120)
        y_data = np.exp(x_data)

        analysis = analyze(x_data, y_data)

        self.assertTrue(analysis["should_attempt_regression"])
        self.assertEqual(analysis["recommended_mode"], "instant")
        self.assertIn("looks_exponential", analysis["flags"])
        self.assertGreaterEqual(len(analysis["likely_families"]), 1)
        self.assertEqual(analysis["feature_names"], ["x"])

    def test_regress_returns_structured_llm_metadata(self):
        x_data = np.linspace(0.2, 4.0, 120)
        y_data = np.exp(x_data)

        result = regress(x_data, y_data, verbose=False)

        self.assertEqual(result["quality"], "exact")
        self.assertEqual(result["confidence"]["label"], "high")
        self.assertEqual(result["verification"]["status"], "verified")
        self.assertIsInstance(result["analysis"], dict)
        self.assertIsInstance(result["candidates"], list)
        self.assertGreaterEqual(len(result["candidates"]), 1)
        self.assertIn("why_best", result["guidance"])
        self.assertEqual(result["analysis"]["feature_names"], ["x"])

    def test_auto_mode_uses_analysis_recommendation(self):
        x_data = np.linspace(0.2, 4.0, 120)
        y_data = np.exp(x_data)

        result = regress(x_data, y_data, verbose=False, mode="auto")

        self.assertEqual(result["requested_mode"], "auto")
        self.assertEqual(result["effective_mode"], "instant")
        self.assertEqual(result["search_budget"]["max_depth"], 2)
        self.assertEqual(result["analysis"]["recommended_mode"], "instant")

    def test_explicit_mode_overrides_default_budget(self):
        x_data = np.linspace(0.2, 4.0, 120)
        y_data = np.exp(x_data)

        result = regress(x_data, y_data, verbose=False, mode="deep")

        self.assertEqual(result["requested_mode"], "deep")
        self.assertEqual(result["effective_mode"], "deep")
        self.assertEqual(result["search_budget"]["max_depth"], 4)
        self.assertEqual(result["search_budget"]["max_random"], 30000)

    def test_two_variable_tree_search_finds_direct_eml_form(self):
        x0 = np.linspace(0.1, 1.2, 80)
        x1 = np.linspace(1.1, 2.8, 80)
        x_matrix = np.column_stack([x0, x1])
        y_data = np.exp(x0) - np.log(x1)

        result = regress(x_matrix, y_data, max_depth=1, verbose=False, workers=1, seed=123)

        self.assertLess(result["mse"], 1e-8)
        self.assertEqual(result["strategy"], "eml_tree")
        self.assertEqual(result["leaf_types"], ["x0", "x1"])
        self.assertEqual(result["used_features"], ["x0", "x1"])

    def test_fixed_seed_makes_search_reproducible(self):
        x_data = np.linspace(0.1, 4.0, 40)
        y_data = np.sin(x_data)

        result_a = regress(x_data, y_data, max_depth=1, verbose=False, workers=1, seed=1234)
        result_b = regress(x_data, y_data, max_depth=1, verbose=False, workers=1, seed=1234)

        self.assertEqual(result_a["leaf_types"], result_b["leaf_types"])
        self.assertEqual(result_a["expression"], result_b["expression"])
        self.assertEqual(result_a["strategy"], result_b["strategy"])
        self.assertAlmostEqual(result_a["mse"], result_b["mse"])
        np.testing.assert_allclose(result_a["constants"], result_b["constants"], rtol=0, atol=1e-12)

    def test_regress_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            regress([], [], verbose=False)
        with self.assertRaises(ValueError):
            regress([1.0, 2.0], [1.0], verbose=False)
        with self.assertRaises(ValueError):
            regress([1.0, np.nan], [1.0, 2.0], verbose=False)
        with self.assertRaises(ValueError):
            regress(np.ones((10, 4)), np.ones(10), verbose=False)
        with self.assertRaises(ValueError):
            regress(np.ones((10, 2)), np.ones(10), verbose=False, feature_names=["only_one"])
        with self.assertRaises(ValueError):
            regress([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], verbose=False, mode="turbo")


if __name__ == "__main__":
    unittest.main()
