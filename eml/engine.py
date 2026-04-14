#!/usr/bin/env python3
"""
EML Symbolic Regression Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core search engine. Uses numpy/scipy for speed, sympy for simplification.

Based on:
    Odrzywołek, A. (2026). "All elementary functions from a single binary
    operator." arXiv:2603.21852

The EML operator:  eml(a, b) = exp(a) - ln(b)
Grammar:           S -> x | 0 | 1 | c | eml(S, S)
"""

import numpy as np
from scipy.optimize import minimize
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import sympy as sp
import json
import sys
import os
import argparse
import warnings
import time

warnings.filterwarnings("ignore")


# ── Leaf alphabet ───────────────────────────────────────────────────

X    = "x"   # input variable
ONE  = "1"   # constant 1
ZERO = "0"   # constant 0
C    = "c"   # trainable constant (fitted to data)

LEAF_OPTIONS = [X, ONE, ZERO, C]
EXHAUSTIVE_LIMIT = 20000
MIN_VALID_FRACTION = 0.9


# ── Core EML operator ──────────────────────────────────────────────

def eml_np(a, b):
    """eml(a, b) = exp(a) - ln(|b|).

    Numerically stabilised: clamps exp input to [-50, 50],
    clamps ln input away from zero.
    """
    with np.errstate(all="ignore"):
        return np.exp(np.clip(a, -50, 50)) - np.log(np.maximum(np.abs(b), 1e-300))


def _x_symbol_for_data(x_data):
    """Pick the strongest safe assumption for x based on the observed domain."""
    if len(x_data) and np.all(x_data > 0):
        return sp.Symbol("x", positive=True, real=True)
    if len(x_data) and np.all(x_data < 0):
        return sp.Symbol("x", negative=True, real=True)
    return sp.Symbol("x", real=True)


def _score_prediction(pred, y_data, min_valid_fraction=MIN_VALID_FRACTION):
    """Mean squared error with a penalty for partially invalid predictions."""
    mask = np.isfinite(pred)
    valid = int(mask.sum())
    required = max(1, int(np.ceil(len(y_data) * min_valid_fraction)))
    if valid < required:
        return float("inf")
    raw_mse = float(np.mean((pred[mask] - y_data[mask]) ** 2))
    return raw_mse * (len(y_data) / valid)


def _build_result(expression, mse, depth, leaf_types=None, constants=None, eml_expression=None):
    """Normalise result payloads across search and pre-pass paths."""
    return {
        "expression": str(expression) if expression is not None else None,
        "eml_expression": eml_expression,
        "mse": float(mse),
        "depth": depth,
        "constants": list(constants or []),
        "leaf_types": list(leaf_types) if leaf_types else [],
    }


def _fingerprint_array(values, sample_idx):
    """Compact numeric signature for deduplicating constant-free trees."""
    sample = values[sample_idx]
    if not np.all(np.isfinite(sample)):
        return None
    return tuple(np.round(sample, 8))


def _evaluate_constant_free_tree(leaf_types, cache):
    """Evaluate a no-constant tree with memoised subtrees."""
    key = tuple(leaf_types)
    if key in cache:
        return cache[key]
    mid = len(key) // 2
    value = eml_np(
        _evaluate_constant_free_tree(key[:mid], cache),
        _evaluate_constant_free_tree(key[mid:], cache),
    )
    cache[key] = value
    return value


def _should_prune_config(cfg):
    """Skip trees with constant-only substructures that collapse to simpler forms."""
    for i in range(0, len(cfg), 2):
        pair = cfg[i:i + 2]
        if X not in pair and pair.count(C) > 1:
            return True

    if len(cfg) >= 4:
        for i in range(0, len(cfg), 4):
            group = cfg[i:i + 4]
            if X not in group and C in group:
                return True
    return False


# ── Tree evaluation ─────────────────────────────────────────────────

def evaluate_tree(leaf_types, x_data, constants=None):
    """Evaluate a full binary EML tree bottom-up. Pure numpy."""
    c_idx = 0
    leaves = []
    for lt in leaf_types:
        if lt == X:
            leaves.append(x_data)
        elif lt == ONE:
            leaves.append(np.ones_like(x_data))
        elif lt == ZERO:
            leaves.append(np.zeros_like(x_data))
        else:  # C
            leaves.append(np.full_like(x_data, float(constants[c_idx])))
            c_idx += 1

    cur = leaves
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            nxt.append(eml_np(cur[i], cur[i + 1]))
        cur = nxt
    return cur[0]


# ── Constant optimisation ──────────────────────────────────────────

def optimize_constants(leaf_types, x_data, y_data, n_constants,
                       restarts=3, quick=False):
    """Fit trainable constants via Nelder-Mead.

    ~7ms per call (quick), ~15ms (full). Near-instant for 1-4 params.
    """
    if quick:
        restarts = 1
        maxiter = max(200, 150 * n_constants)
    else:
        maxiter = max(2000, 500 * n_constants)

    rng = np.random.default_rng()
    base_scales = [0.5, 2.0, 10.0]

    def objective(c):
        pred = evaluate_tree(leaf_types, x_data, c)
        mse = _score_prediction(pred, y_data)
        return mse if np.isfinite(mse) else 1e20

    best_mse, best_c = float("inf"), None
    for r in range(restarts):
        if r < len(base_scales):
            scale = base_scales[r]
        else:
            scale = 10 ** rng.uniform(-1.0, 1.0)
        c0 = rng.normal(size=n_constants) * scale
        try:
            res = minimize(objective, c0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-12, "fatol": 1e-12})
            if res.fun < best_mse:
                best_mse = res.fun
                best_c = res.x.tolist()
        except Exception:
            continue
    return best_mse, best_c


# ── Sympy conversion ───────────────────────────────────────────────

_KNOWN = None

def _known_constants():
    """Table of common constants for pretty-printing."""
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = [
            (0, sp.Integer(0)),       (1, sp.Integer(1)),
            (-1, sp.Integer(-1)),     (2, sp.Integer(2)),
            (-2, sp.Integer(-2)),     (0.5, sp.Rational(1, 2)),
            (-0.5, sp.Rational(-1, 2)),
            (1/3, sp.Rational(1, 3)), (2/3, sp.Rational(2, 3)),
            (np.pi, sp.pi),           (-np.pi, -sp.pi),
            (np.pi/2, sp.pi/2),       (np.pi/4, sp.pi/4),
            (np.e, sp.E),             (np.log(2), sp.log(2)),
            (np.sqrt(2), sp.sqrt(2)), (np.sqrt(3), sp.sqrt(3)),
        ]
    return _KNOWN


def _nice(val, tol=1e-5):
    """Map a float to a clean sympy constant when possible."""
    for cv, sym in _known_constants():
        threshold = min(tol, 1e-10) if cv == 0 else tol
        if abs(val - cv) < threshold:
            return sym
    try:
        candidate = sp.nsimplify(
            val,
            constants=[sp.pi, sp.E, sp.log(2), sp.sqrt(2), sp.sqrt(3)],
            tolerance=tol,
        )
        if candidate.is_number:
            approx = float(sp.N(candidate, 16))
            if abs(approx - val) < tol and sp.count_ops(candidate) <= 3:
                return candidate
    except Exception:
        pass
    return sp.Float(val, 4)


def _fit_standard_forms(x_data, y_data, tolerance):
    """Cheap exact-fit pre-pass for common one-variable function families."""
    x_sym = _x_symbol_for_data(x_data)
    nice_tol = max(tolerance, 1e-5)

    def _maybe_result(pred, expr):
        mse = _score_prediction(pred, y_data, min_valid_fraction=1.0)
        if mse < tolerance:
            return _build_result(sp.simplify(expr), mse, depth=0)
        return None

    for degree in range(1, 5):
        if len(x_data) < degree + 1:
            continue
        coeffs = np.polyfit(x_data, y_data, degree)
        pred = np.polyval(coeffs, x_data)
        expr = sp.Integer(0)
        for power, coeff in enumerate(reversed(coeffs)):
            if abs(coeff) < nice_tol:
                continue
            expr += _nice(float(coeff), tol=nice_tol) * x_sym ** power
        result = _maybe_result(pred, sp.expand(expr))
        if result is not None:
            return result

    if np.all(x_data > 0) and np.all(y_data > 0):
        slope, intercept = np.polyfit(np.log(x_data), np.log(y_data), 1)
        scale = float(np.exp(intercept))
        pred = scale * np.power(x_data, slope)
        expr = _nice(scale, tol=nice_tol) * x_sym ** _nice(slope, tol=nice_tol)
        result = _maybe_result(pred, expr)
        if result is not None:
            return result

    if np.all(y_data > 0):
        slope, intercept = np.polyfit(x_data, np.log(y_data), 1)
        scale = float(np.exp(intercept))
        pred = scale * np.exp(slope * x_data)
        expr = _nice(scale, tol=nice_tol) * sp.exp(_nice(slope, tol=nice_tol) * x_sym)
        result = _maybe_result(pred, expr)
        if result is not None:
            return result

    if np.all(x_data > 0):
        slope, intercept = np.polyfit(np.log(x_data), y_data, 1)
        pred = slope * np.log(x_data) + intercept
        expr = _nice(slope, tol=nice_tol) * sp.log(x_sym) + _nice(intercept, tol=nice_tol)
        result = _maybe_result(pred, expr)
        if result is not None:
            return result
    return None


def tree_to_sympy(leaf_types, constants=None, x_symbol=None):
    """Convert an EML tree to a simplified sympy expression."""
    x_sym = x_symbol if x_symbol is not None else sp.Symbol("x", real=True)
    c_idx = 0
    leaves = []
    for lt in leaf_types:
        if lt == X:
            leaves.append(x_sym)
        elif lt == ZERO:
            leaves.append(sp.Integer(0))
        elif lt == ONE:
            leaves.append(sp.Integer(1))
        else:
            v = constants[c_idx] if constants else sp.Symbol(f"c{c_idx}")
            if isinstance(v, (int, float)):
                v = _nice(v)
            leaves.append(v)
            c_idx += 1

    cur = leaves
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i + 1]
            if b.is_zero is True:
                return None
            nxt.append(sp.exp(a) - sp.log(sp.Abs(b)))
        cur = nxt

    result = cur[0]
    try:
        result = sp.simplify(result)
    except Exception:
        pass
    if result.has(sp.nan, sp.zoo, sp.oo, sp.S.NegativeInfinity, sp.I, sp.Symbol("eps")):
        return None
    return result


def _eml_str(leaf_types, constants=None):
    """Build a readable EML-tree string."""
    c_idx = 0
    strs = []
    for lt in leaf_types:
        if lt == X:
            strs.append("x")
        elif lt == ONE:
            strs.append("1")
        elif lt == ZERO:
            strs.append("0")
        else:
            v = constants[c_idx] if constants and c_idx < len(constants) else f"c{c_idx}"
            strs.append(str(_nice(v)) if isinstance(v, (int, float)) else str(v))
            c_idx += 1
    while len(strs) > 1:
        nxt = []
        for i in range(0, len(strs), 2):
            nxt.append(f"eml({strs[i]}, {strs[i + 1]})")
        strs = nxt
    return strs[0]


# ── Search engine ───────────────────────────────────────────────────

def _screen_one(args):
    """Worker for parallel Phase 2 screening."""
    lt, x_data, y_data, nc = args
    mse, consts = optimize_constants(lt, x_data, y_data, nc, quick=True)
    return mse, lt, nc, consts


def symbolic_regression(x_data, y_data, max_depth=3, tolerance=1e-8,
                        verbose=True, max_random=10000, workers=None):
    """Search for the simplest EML tree that fits the data.

    A cheap exact-form pre-pass runs before the EML search. If that misses,
    each depth level uses three phases:
      1) Instant eval of constant-free trees (pure numpy)
      2) Parallel quick-screen of trees with trainable constants
      3) Full optimisation of top candidates

    Args:
        x_data:     Input array.
        y_data:     Target array.
        max_depth:  Max tree depth (1-4). 3 recommended.
        tolerance:  Early-stop MSE threshold.
        verbose:    Print progress.
        max_random: Sample count when exhaustive search is too large.
        workers:    Parallel workers (default: auto).

    Returns:
        dict with expression, eml_expression, mse, depth, constants, leaf_types.
    """
    x_data = np.asarray(x_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)
    if x_data.ndim != 1 or y_data.ndim != 1:
        raise ValueError("x_data and y_data must be 1D arrays")
    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("x_data and y_data must be non-empty")
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
        raise ValueError("x_data and y_data must contain only finite values")
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)

    best = _build_result(None, float("inf"), depth=None)
    t0 = time.time()
    x_symbol = _x_symbol_for_data(x_data)

    def _update(lt, mse, consts, depth):
        if mse < best["mse"]:
            try:
                sym = tree_to_sympy(lt, consts or None, x_symbol=x_symbol)
            except Exception:
                sym = None
            best.update(_build_result(
                sym if sym is not None else _eml_str(lt, consts),
                mse,
                depth,
                leaf_types=lt,
                constants=consts,
                eml_expression=_eml_str(lt, consts),
            ))
            if verbose and mse < 1.0:
                print(f"  new best  MSE={mse:.2e}  {best['expression']}")
            return mse < tolerance
        return False

    prepass = _fit_standard_forms(x_data, y_data, tolerance)
    if prepass is not None:
        if verbose:
            print("[pre-pass] exact fit found")
            _report(prepass, time.time() - t0)
        return prepass

    for depth in range(1, max_depth + 1):
        n_leaves = 2 ** depth
        n_cfgs = len(LEAF_OPTIONS) ** n_leaves
        exhaustive = n_cfgs <= EXHAUSTIVE_LIMIT

        if exhaustive:
            all_cfgs = list(product(LEAF_OPTIONS, repeat=n_leaves))
        else:
            all_cfgs = [tuple(np.random.choice(LEAF_OPTIONS, n_leaves))
                        for _ in range(max_random)]

        all_cfgs = [c for c in all_cfgs if X in c]

        if verbose:
            mode = "exhaustive" if exhaustive else f"sampling {max_random}"
            print(f"[depth {depth}] {n_leaves} leaves | {mode} | {len(all_cfgs)} viable")

        # Phase 1: constant-free trees (instant)
        no_const = [c for c in all_cfgs if C not in c]
        leaf_cache = {
            (X,): x_data,
            (ONE,): np.ones_like(x_data),
            (ZERO,): np.zeros_like(x_data),
        }
        sample_idx = np.linspace(0, len(x_data) - 1, num=min(5, len(x_data)), dtype=int)
        seen_fingerprints = set()
        unique_no_const = []
        for cfg in no_const:
            lt = list(cfg)
            pred = _evaluate_constant_free_tree(lt, leaf_cache)
            fp = _fingerprint_array(pred, sample_idx)
            if fp is None or fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            unique_no_const.append(cfg)
            mse = _score_prediction(pred, y_data)
            if _update(lt, mse, [], depth):
                if verbose:
                    _report(best, time.time() - t0)
                return best

        if verbose:
            print(f"  phase 1  ({len(unique_no_const):>5} unique trees, no constants): "
                  f"best MSE = {best['mse']:.2e}")

        # Phase 2: parallel quick screen
        has_const = [c for c in all_cfgs if C in c and not _should_prune_config(c)]
        tasks = [(list(cfg), x_data, y_data, list(cfg).count(C))
                 for cfg in has_const]
        candidates = []

        if len(tasks) > 100:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for mse, lt, nc, consts in pool.map(_screen_one, tasks, chunksize=64):
                    if consts is None:
                        continue
                    candidates.append((mse, lt, nc, consts))
                    _update(lt, mse, consts, depth)
        else:
            for t in tasks:
                mse, lt, nc, consts = _screen_one(t)
                if consts is None:
                    continue
                candidates.append((mse, lt, nc, consts))
                if _update(lt, mse, consts, depth) and mse < tolerance:
                    if verbose:
                        _report(best, time.time() - t0)
                    return best

        if verbose:
            print(f"  phase 2  ({len(has_const):>5} trees, screened):    "
                  f"best MSE = {best['mse']:.2e}")

        if best["mse"] < tolerance:
            if verbose:
                _report(best, time.time() - t0)
            return best

        # Phase 3: full optimisation on top candidates
        candidates.sort(key=lambda c: c[0])
        for _, lt, nc, _ in candidates[:30]:
            mse, consts = optimize_constants(lt, x_data, y_data, nc)
            if consts is None:
                continue
            if _update(lt, mse, consts, depth) and mse < tolerance:
                if verbose:
                    _report(best, time.time() - t0)
                return best

        if verbose:
            print(f"  phase 3  (top 30, full optimise):   "
                  f"best MSE = {best['mse']:.2e}")

    if verbose:
        _report(best, time.time() - t0)
    return best


def _report(b, elapsed):
    print(f"\n  {'='*50}")
    print(f"  Formula : {b.get('expression') or 'none'}")
    print(f"  EML tree: {b.get('eml_expression') or 'n/a'}")
    print(f"  MSE     : {b['mse']:.2e}")
    depth = b.get("depth")
    depth_label = depth if depth is not None else "n/a"
    print(f"  Depth   : {depth_label}   ({elapsed:.1f}s)")
    print(f"  {'='*50}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    """CLI entry point (also available as `eml-regress` after pip install)."""
    ap = argparse.ArgumentParser(
        prog="eml-regress",
        description="Discover mathematical formulas from data using EML trees")
    ap.add_argument("--demo", action="store_true",
                    help="Run demo on common functions")
    ap.add_argument("--json", type=str,
                    help='JSON data file: {"x":[...], "y":[...]}')
    ap.add_argument("--func", type=str,
                    help='Generate data from expression, e.g. "sin(x)"')
    ap.add_argument("--range", nargs=2, type=float, default=[0.1, 5.0],
                    metavar=("LO", "HI"),
                    help="x range for --func (default: 0.1 5.0)")
    ap.add_argument("--points", type=int, default=200,
                    help="Number of data points (default: 200)")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="Maximum tree depth 1-4 (default: 3)")
    ap.add_argument("--tolerance", type=float, default=1e-8,
                    help="MSE tolerance for early stop (default: 1e-8)")
    ap.add_argument("--output-json", action="store_true",
                    help="Output result as JSON")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress progress output")
    args = ap.parse_args()

    if args.demo:
        _run_demo()
        return

    if args.json:
        with open(args.json) as f:
            d = json.load(f)
        xd, yd = np.array(d["x"], float), np.array(d["y"], float)
    elif args.func:
        xd = np.linspace(args.range[0], args.range[1], args.points)
        yd = sp.lambdify(sp.Symbol("x"), sp.sympify(args.func), "numpy")(xd)
    elif not sys.stdin.isatty():
        d = json.load(sys.stdin)
        xd, yd = np.array(d["x"], float), np.array(d["y"], float)
        args.max_depth = d.get("max_depth", args.max_depth)
        args.output_json = True
        args.quiet = True
    else:
        ap.print_help()
        return

    r = symbolic_regression(xd, yd, max_depth=args.max_depth,
                            tolerance=args.tolerance, verbose=not args.quiet)
    if args.output_json:
        print(json.dumps({
            "expression": r["expression"],
            "eml_expression": r["eml_expression"],
            "depth": r["depth"],
            "mse": r["mse"],
            "constants": r["constants"],
            "leaf_types": r["leaf_types"],
        }, indent=2))


def _run_demo():
    x = np.linspace(0.1, 4.0, 200)
    demos = [
        ("exp(x)",  np.exp(x)),
        ("1/x",     1.0 / x),
        ("ln(x)",   np.log(x)),
        ("x + 1",   x + 1),
        ("x^2",     x ** 2),
        ("2*x",     2 * x),
    ]
    print()
    print("  EML SYMBOLIC REGRESSION")
    print("  eml(a, b) = exp(a) - ln(b)")
    print()
    for name, y in demos:
        print(f"  --- target: {name} ---")
        r = symbolic_regression(x, y, max_depth=3)
        status = "EXACT" if r["mse"] < 1e-8 else f"approx (MSE {r['mse']:.2e})"
        print(f"  => {r['expression']}   [{status}]\n")


if __name__ == "__main__":
    main()
