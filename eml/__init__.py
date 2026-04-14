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

from .engine import symbolic_regression, evaluate_tree, eml_np

__version__ = "0.1.0"
__all__ = ["regress", "symbolic_regression", "evaluate_tree", "eml_np"]


def regress(x, y, max_depth=3, tolerance=1e-8, verbose=False, workers=None):
    """Discover the formula behind your data.

    Args:
        x: Input values (array-like, 1D).
        y: Output values (array-like, same length as x).
        max_depth: Maximum EML tree depth (1-4). Default 3.
        tolerance: Stop early if MSE drops below this. Default 1e-8.
        verbose: Print search progress. Default False.
        workers: Number of parallel workers. Default auto.

    Returns:
        Result dict with keys:
            expression     - Simplified formula string (e.g. "exp(x)")
            eml_expression - Raw EML tree (e.g. "eml(x, 1)")
            mse            - Mean squared error of fit
            depth          - Tree depth used
            constants      - Optimised constant values
            leaf_types     - Leaf configuration of best tree
    """
    return symbolic_regression(
        x, y,
        max_depth=max_depth,
        tolerance=tolerance,
        verbose=verbose,
        workers=workers,
        max_random=10000,
    )
