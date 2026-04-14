"""
EML Symbolic Regression
~~~~~~~~~~~~~~~~~~~~~~~

Discover exact mathematical formulas from data using a single operator.

    eml(a, b) = exp(a) - ln(b)

Basic usage:

    >>> from eml import regress
    >>> import numpy as np
    >>> x = np.linspace(0.1, 5, 200)
    >>> result = regress(x, np.exp(x))
    >>> result.expression
    'exp(x)'

Full docs: https://github.com/Aezaror/eml-symbolic-regression
"""

from .engine import analyze_dataset, symbolic_regression, evaluate_tree, eml_np

__version__ = "0.1.0"
__all__ = ["analyze", "regress", "symbolic_regression", "evaluate_tree", "eml_np"]


def analyze(x, y, feature_names=None):
    """Classify whether symbolic regression is worth attempting on a dataset."""
    return analyze_dataset(x, y, feature_names=feature_names)


def regress(x, y, max_depth=None, tolerance=1e-8, verbose=False, workers=None,
            seed=None, feature_names=None, mode=None):
    """Discover the formula behind your data.

    Args:
        x: Input values (array-like, 1D) or feature matrix (array-like, 2D).
        y: Output values (array-like, same length as x).
        max_depth: Optional explicit maximum EML tree depth override.
        tolerance: Stop early if MSE drops below this. Default 1e-8.
        verbose: Print search progress. Default False.
        workers: Number of parallel workers. Default auto.
        seed: Optional RNG seed for deterministic search.
        feature_names: Optional names for multivariable inputs. Defaults to
            `x0`, `x1`, ... for matrices, while legacy 1D vectors still use `x`.
        mode: Search preset: `instant`, `balanced`, `deep`, `research`, or `auto`.

    Returns:
        Result dict with keys:
            expression     - Simplified formula string (e.g. "exp(x)")
            eml_expression - Raw EML tree (e.g. "eml(x, 1)")
            mse            - Mean squared error of fit
            depth          - Tree depth used
            constants      - Optimised constant values
            leaf_types     - Leaf configuration of best tree
            feature_names  - Resolved feature names used by the engine
            used_features  - Which input features appear in the final formula
            analysis       - Routing diagnostics before the search
            verification   - Post-search fit checks and failure reporting
            confidence     - Compact trust score for LLM agents
            candidates     - Ranked alternative formulas
            guidance       - LLM-oriented explanation scaffolding
            requested_mode - Mode requested by the caller
            effective_mode - Mode actually used after auto-routing
            search_budget  - Concrete max_depth / max_random budget
    """
    return symbolic_regression(
        x, y,
        max_depth=max_depth,
        tolerance=tolerance,
        verbose=verbose,
        workers=workers,
        seed=seed,
        feature_names=feature_names,
        mode=mode,
    )
