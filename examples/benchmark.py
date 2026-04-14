"""
Benchmark: test EML regression against common functions.
Shows what works well, what's harder, and timing.
"""
from eml import regress
import numpy as np
import time


def main():
    x = np.linspace(0.1, 4.0, 200)

    targets = [
        ("exp(x)",       np.exp(x),     2),
        ("1/x",          1.0 / x,       2),
        ("ln(x)",        np.log(x),     3),
        ("exp(-x)",      np.exp(-x),    2),
        ("x + 1",        x + 1,         3),
        ("x^2",          x ** 2,        3),
        ("sqrt(x)",      np.sqrt(x),    3),
        ("sin(x)",       np.sin(x),     3),
    ]

    print(f"{'Function':<14} {'Found':<30} {'MSE':>10} {'Depth':>6} {'Time':>7}")
    print("-" * 72)

    for name, y, md in targets:
        t0 = time.time()
        r = regress(x, y, max_depth=md)
        dt = time.time() - t0

        expr = r["expression"]
        if len(expr) > 28:
            expr = expr[:25] + "..."

        tag = "EXACT" if r["mse"] < 1e-8 else ""
        print(f"{name:<14} {expr:<30} {r['mse']:>10.2e} {r['depth']:>5}d {dt:>6.1f}s  {tag}")


if __name__ == "__main__":
    main()
