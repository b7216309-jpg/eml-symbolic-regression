import unittest

import numpy as np
import sympy as sp

from eml import regress
from eml.engine import C, ONE, X, ZERO, evaluate_tree, tree_to_sympy


def _eval_sympy(expr, x_data):
    if expr is None:
        raise AssertionError("Expression should not be None")
    if not expr.free_symbols:
        return np.full_like(x_data, float(expr), dtype=np.float64)
    x_sym = next(iter(expr.free_symbols))
    fn = sp.lambdify(x_sym, expr, "numpy")
    return np.asarray(fn(x_data), dtype=np.float64)


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

    def test_symbolic_conversion_matches_numeric_tree_evaluation(self):
        x_data = np.linspace(0.1, 3.0, 64)
        x_symbol = sp.Symbol("x", positive=True, real=True)
        cases = [
            ([X, ONE], []),
            ([ONE, X], []),
            ([X, C], [-2.0]),
            ([ONE, C, X, ONE], [-2.0]),
        ]

        for leaf_types, constants in cases:
            with self.subTest(leaf_types=leaf_types, constants=constants):
                expr = tree_to_sympy(leaf_types, constants=constants, x_symbol=x_symbol)
                self.assertIsNotNone(expr)
                np.testing.assert_allclose(
                    _eval_sympy(expr, x_data),
                    evaluate_tree(leaf_types, x_data, constants),
                    rtol=1e-9,
                    atol=1e-9,
                )


class RegressionTests(unittest.TestCase):
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
                for banned in ("eps", "I*pi", "zoo", "nan"):
                    self.assertNotIn(banned, result["expression"] or "")

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


if __name__ == "__main__":
    unittest.main()
