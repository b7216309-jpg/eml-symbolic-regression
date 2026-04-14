"""
Benchmark: test EML regression against common multivariable functions.
Shows what the multivariable pre-pass solves immediately, what falls to
EML tree search, and how long each case takes.
"""
from eml import regress
import numpy as np
import time


def make_grid(x0_lo=0.1, x0_hi=2.0, x1_lo=0.5, x1_hi=2.5, n0=12, n1=10):
    """Create a 2D feature grid flattened into an (n_samples, 2) matrix."""
    x0, x1 = np.meshgrid(
        np.linspace(x0_lo, x0_hi, n0),
        np.linspace(x1_lo, x1_hi, n1),
    )
    return np.column_stack([x0.ravel(), x1.ravel()])


def make_grid3(a_lo=1.0, a_hi=4.0, b_lo=2.0, b_hi=5.0, c_lo=1.0, c_hi=3.0, na=8, nb=7, nc=6):
    """Create a 3D feature grid flattened into an (n_samples, 3) matrix."""
    a, b, c = np.meshgrid(
        np.linspace(a_lo, a_hi, na),
        np.linspace(b_lo, b_hi, nb),
        np.linspace(c_lo, c_hi, nc),
    )
    return np.column_stack([a.ravel(), b.ravel(), c.ravel()])


def main():
    X = make_grid()
    x0 = X[:, 0]
    x1 = X[:, 1]
    X3 = make_grid3()
    m1 = X3[:, 0]
    m2 = X3[:, 1]
    r = X3[:, 2]

    targets = [
        ("x0 + x1",        X,  x0 + x1,                         2, None),
        ("x0 * x1",        X,  x0 * x1,                         2, None),
        ("x0 / x1",        X,  x0 / x1,                         2, None),
        ("exp(x0 + x1)",   X,  np.exp(x0 + x1),                 2, None),
        ("x0^2 + x1",      X,  x0 ** 2 + x1,                    2, None),
        ("eml(x0, x1)",    X,  np.exp(x0) - np.log(x1),         1, None),
        ("m1*m2/r^2",      X3, (m1 * m2) / (r ** 2),            3, ["m1", "m2", "r"]),
    ]

    print(
        f"{'Function':<16} {'Found':<28} {'MSE':>10} "
        f"{'Strategy':<10} {'Depth':>6} {'Used':<10} {'Time':>7}"
    )
    print("-" * 98)

    for name, features, y, max_depth, feature_names in targets:
        t0 = time.time()
        result = regress(
            features,
            y,
            max_depth=max_depth,
            verbose=False,
            feature_names=feature_names,
        )
        dt = time.time() - t0

        expr = result["expression"] or "none"
        if len(expr) > 26:
            expr = expr[:23] + "..."

        used = ",".join(result.get("used_features", [])) or "-"
        print(
            f"{name:<16} {expr:<28} {result['mse']:>10.2e} "
            f"{result['strategy']:<10} {result['depth']:>5}d {used:<10} {dt:>6.1f}s"
        )

    print()
    print("Current search supports up to 3 input features.")
    print("Pre-pass hits return at depth 0; direct EML-tree hits report their actual tree depth.")


if __name__ == "__main__":
    main()
