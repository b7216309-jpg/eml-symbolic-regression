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

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import basinhopping, minimize
import sympy as sp

warnings.filterwarnings("ignore")


# ── Leaf alphabet ───────────────────────────────────────────────────

X = "x"      # input variable for legacy 1D vector calls
ONE = "1"    # constant 1
ZERO = "0"   # constant 0
C = "c"      # trainable constant (fitted to data)

LEAF_OPTIONS = [X, ONE, ZERO, C]
MAX_SEARCH_FEATURES = 3
EXHAUSTIVE_LIMIT = 20000
MIN_VALID_FRACTION = 0.9
SEARCH_MODE_PRESETS = {
    "instant": {"max_depth": 2, "max_random": 1500},
    "balanced": {"max_depth": 3, "max_random": 10000},
    "deep": {"max_depth": 4, "max_random": 30000},
    "research": {"max_depth": 5, "max_random": 60000},
}
SAMPLE_BUCKET_WEIGHTS = {
    0: 0.10,
    1: 0.30,
    2: 0.30,
    3: 0.20,
    4: 0.10,
}

_WORKER_X_DATA = None
_WORKER_Y_DATA = None


# ── Core EML operator ──────────────────────────────────────────────

def eml_np(a, b):
    """eml(a, b) = exp(a) - ln(|b|).

    Numerically stabilised: clamps exp input to [-50, 50],
    clamps ln input away from zero.
    """
    with np.errstate(all="ignore"):
        return np.exp(np.clip(a, -50, 50)) - np.log(np.maximum(np.abs(b), 1e-300))


def _is_feature_token(token):
    """Return True when the token denotes an input feature leaf."""
    return isinstance(token, str) and token not in {ONE, ZERO, C}


def _contains_feature(cfg):
    """Require each candidate tree to depend on at least one input feature."""
    return any(_is_feature_token(token) for token in cfg)


def _validate_feature_names(feature_names, n_features):
    """Validate user-supplied feature names."""
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match input width ({n_features})"
        )
    seen = set()
    for name in feature_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("feature_names must contain non-empty strings")
        if name in {ONE, ZERO, C}:
            raise ValueError(f"feature name '{name}' is reserved")
        if name in seen:
            raise ValueError("feature_names must be unique")
        seen.add(name)
    return [str(name) for name in feature_names]


def _default_feature_names(n_features, preserve_scalar_name):
    """Pick default feature names for 1D vectors vs. explicit matrices."""
    if n_features == 1 and preserve_scalar_name:
        return [X]
    return [f"x{i}" for i in range(n_features)]


def _coerce_feature_matrix(x_data, feature_names=None, *, preserve_scalar_name=True):
    """Normalise vector/matrix inputs into a feature matrix plus names."""
    x_arr = np.asarray(x_data, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
        default_names = _default_feature_names(1, preserve_scalar_name)
    elif x_arr.ndim == 2:
        default_names = _default_feature_names(x_arr.shape[1], preserve_scalar_name=False)
    else:
        raise ValueError("x_data must be a 1D array or 2D feature matrix")

    if x_arr.shape[0] == 0 or x_arr.shape[1] == 0:
        raise ValueError("x_data must be non-empty")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x_data must contain only finite values")

    if feature_names is None:
        feature_names = default_names
    else:
        feature_names = _validate_feature_names(list(feature_names), x_arr.shape[1])
    return x_arr, list(feature_names)


def _symbol_for_feature(name, values):
    """Pick the strongest safe assumption for one feature based on its domain."""
    if len(values) and np.all(values > 0):
        return sp.Symbol(name, positive=True, real=True)
    if len(values) and np.all(values < 0):
        return sp.Symbol(name, negative=True, real=True)
    return sp.Symbol(name, real=True)


def _feature_symbol_map(x_data, feature_names):
    """Create a stable feature-name -> sympy symbol map."""
    return {
        name: _symbol_for_feature(name, x_data[:, idx])
        for idx, name in enumerate(feature_names)
    }


def _used_features_from_leaf_types(leaf_types, feature_names):
    """Preserve feature order in result metadata."""
    present = set(token for token in leaf_types if _is_feature_token(token))
    return [name for name in feature_names if name in present]


def _score_prediction(pred, y_data, min_valid_fraction=MIN_VALID_FRACTION):
    """Mean squared error with a penalty for partially invalid predictions."""
    mask = np.isfinite(pred)
    valid = int(mask.sum())
    required = max(1, int(np.ceil(len(y_data) * min_valid_fraction)))
    if valid < required:
        return float("inf")
    raw_mse = float(np.mean((pred[mask] - y_data[mask]) ** 2))
    return raw_mse * (len(y_data) / valid)


def _coerce_regression_inputs(x_data, y_data, feature_names=None):
    """Validate and normalise regression inputs once for all entry points."""
    x_data, feature_names = _coerce_feature_matrix(
        x_data,
        feature_names=feature_names,
        preserve_scalar_name=True,
    )
    y_data = np.asarray(y_data, dtype=np.float64)
    if y_data.ndim != 1:
        raise ValueError("y_data must be a 1D array")
    if len(y_data) == 0:
        raise ValueError("y_data must be non-empty")
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    if not np.all(np.isfinite(y_data)):
        raise ValueError("y_data must contain only finite values")
    return x_data, y_data, feature_names


def _safe_scale(y_data):
    """Choose a stable normalisation scale for residual metrics."""
    scale = float(np.std(y_data))
    if scale > 1e-12:
        return scale
    scale = float(np.mean(np.abs(y_data)))
    return scale if scale > 1e-12 else 1.0


def _normalized_rmse(mse, y_data):
    """Dimensionless error ratio for heuristic comparisons."""
    if not np.isfinite(mse):
        return float("inf")
    return float(np.sqrt(max(float(mse), 0.0)) / _safe_scale(y_data))


def _safe_corr(a, b):
    """Absolute Pearson correlation, or 0 when degenerate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(abs(corr)) if np.isfinite(corr) else 0.0


def _quality_from_metrics(mse, y_data, tolerance):
    """Map fit quality to a small stable label set."""
    nrmse = _normalized_rmse(mse, y_data)
    if mse < tolerance:
        return "exact"
    if nrmse < 0.10 or mse < max(1e-2, tolerance * 1e5):
        return "approximate"
    return "rough"


def _resolve_search_budget(mode, max_depth, max_random, analysis):
    """Turn a search mode plus optional overrides into concrete limits."""
    requested_mode = mode or "balanced"
    if requested_mode == "auto":
        effective_mode = analysis.get("recommended_mode", "balanced")
    else:
        effective_mode = requested_mode

    if effective_mode not in SEARCH_MODE_PRESETS:
        allowed = ", ".join(["auto"] + list(SEARCH_MODE_PRESETS))
        raise ValueError(f"mode must be one of: {allowed}")

    preset = SEARCH_MODE_PRESETS[effective_mode]
    resolved_max_depth = int(max_depth if max_depth is not None else preset["max_depth"])
    resolved_max_random = int(max_random if max_random is not None else preset["max_random"])
    if resolved_max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if resolved_max_random < 1:
        raise ValueError("max_random must be >= 1")
    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "max_depth": resolved_max_depth,
        "max_random": resolved_max_random,
    }


def _result_complexity(expression, leaf_types=None):
    """Estimate expression complexity for ranking and confidence."""
    if leaf_types:
        return int(len(leaf_types))
    if expression is None:
        return 0
    try:
        return int(sp.count_ops(sp.sympify(expression)))
    except Exception:
        return max(1, len(str(expression)) // 6)


def _empty_candidate_board():
    """Container for deduplicated candidate formulas."""
    return {}


def _candidate_signature(candidate):
    """Stable signature used to deduplicate candidate formulas."""
    return (
        candidate.get("strategy"),
        candidate.get("expression"),
        candidate.get("eml_expression"),
    )


def _build_result(expression, mse, depth, leaf_types=None, constants=None,
                  eml_expression=None, strategy=None, feature_names=None,
                  used_features=None, family=None, rationale=None,
                  quality=None, complexity=None):
    """Normalise result payloads across search and pre-pass paths."""
    return {
        "expression": str(expression) if expression is not None else None,
        "eml_expression": eml_expression,
        "mse": float(mse),
        "depth": depth,
        "constants": list(constants or []),
        "leaf_types": list(leaf_types) if leaf_types else [],
        "strategy": strategy,
        "feature_names": list(feature_names or []),
        "used_features": list(used_features or []),
        "family": family,
        "rationale": rationale,
        "quality": quality,
        "complexity": int(complexity) if complexity is not None else _result_complexity(expression, leaf_types),
        "analysis": {},
        "verification": {},
        "confidence": {},
        "failure_modes": [],
        "candidates": [],
        "guidance": {},
    }


def _register_candidate(board, candidate, max_entries=12):
    """Keep a small ranked set of best distinct candidate formulas."""
    if candidate.get("expression") is None and candidate.get("eml_expression") is None:
        return
    candidate = dict(candidate)
    candidate.pop("analysis", None)
    candidate.pop("verification", None)
    candidate.pop("confidence", None)
    candidate.pop("failure_modes", None)
    candidate.pop("candidates", None)
    candidate.pop("guidance", None)
    key = _candidate_signature(candidate)
    prev = board.get(key)
    rank_key = (
        candidate.get("mse", float("inf")),
        candidate.get("complexity", 999999),
        candidate.get("depth") if candidate.get("depth") is not None else 999999,
    )
    if prev is None:
        board[key] = candidate
    else:
        prev_rank = (
            prev.get("mse", float("inf")),
            prev.get("complexity", 999999),
            prev.get("depth") if prev.get("depth") is not None else 999999,
        )
        if rank_key < prev_rank:
            board[key] = candidate
    if len(board) > max_entries:
        ranked = sorted(
            board.items(),
            key=lambda item: (
                item[1].get("mse", float("inf")),
                item[1].get("complexity", 999999),
                item[1].get("depth") if item[1].get("depth") is not None else 999999,
            ),
        )
        for stale_key, _ in ranked[max_entries:]:
            board.pop(stale_key, None)


def _ranked_candidates(board, limit=5):
    """Return the best candidate list in deterministic order."""
    ranked = sorted(
        board.values(),
        key=lambda item: (
            item.get("mse", float("inf")),
            item.get("complexity", 999999),
            item.get("depth") if item.get("depth") is not None else 999999,
        ),
    )
    result = []
    for rank, candidate in enumerate(ranked[:limit], start=1):
        entry = dict(candidate)
        entry["rank"] = rank
        result.append(entry)
    return result


def _record_family_score(scores, family, pred, y_data, rationale, features=None):
    """Store one cheap family-fit probe for routing heuristics."""
    mse = _score_prediction(pred, y_data, min_valid_fraction=1.0)
    if not np.isfinite(mse):
        return
    scores.append({
        "family": family,
        "mse": float(mse),
        "normalized_rmse": _normalized_rmse(mse, y_data),
        "rationale": rationale,
        "features": list(features or []),
    })


def _periodicity_score(y_data):
    """Detect repeated oscillation without claiming a true symbolic period."""
    if len(y_data) < 12:
        return 0.0
    centred = y_data - np.mean(y_data)
    scale = np.std(centred)
    if scale < 1e-12:
        return 0.0
    acf = np.correlate(centred, centred, mode="full")[len(y_data) - 1:]
    if not len(acf) or acf[0] <= 0:
        return 0.0
    acf = acf / acf[0]
    max_lag = max(3, len(y_data) // 3)
    peak = float(np.max(acf[2:max_lag])) if max_lag > 2 else 0.0
    turns = np.count_nonzero(np.diff(np.sign(np.diff(y_data))) != 0)
    return float(np.clip(max(peak, 0.0) * min(turns / 4.0, 1.0), 0.0, 1.0))


def _piecewise_score(y_data):
    """Crude edge/jump detector for piecewise-looking traces."""
    if len(y_data) < 8:
        return 0.0
    grad = np.diff(y_data)
    if not len(grad):
        return 0.0
    med = float(np.median(np.abs(grad))) + 1e-12
    jump = float(np.max(np.abs(np.diff(grad)))) if len(grad) > 1 else 0.0
    return float(np.clip(jump / (8.0 * med), 0.0, 1.0))


def _feature_importance(x_data, y_data, feature_names):
    """Simple importance proxy for LLM-facing explanations."""
    raw = np.array([_safe_corr(x_data[:, idx], y_data) for idx in range(x_data.shape[1])], dtype=np.float64)
    total = float(raw.sum())
    if total <= 1e-12:
        weights = np.full(len(feature_names), 1.0 / max(1, len(feature_names)), dtype=np.float64)
    else:
        weights = raw / total
    return [
        {"feature": name, "score": float(weight)}
        for name, weight in zip(feature_names, weights)
    ]


def _pairwise_indices(n_features):
    """Enumerate upper-triangular feature pairs once."""
    return [
        (i, j)
        for i in range(n_features)
        for j in range(i + 1, n_features)
    ]


def _multivariate_designs(x_data):
    """Cheap multivariate design matrices for routing and pre-pass fits."""
    ones = np.ones(len(x_data), dtype=np.float64)
    base = [ones] + [x_data[:, idx] for idx in range(x_data.shape[1])]
    pairwise = [x_data[:, i] * x_data[:, j] for i, j in _pairwise_indices(x_data.shape[1])]
    squares = [x_data[:, idx] ** 2 for idx in range(x_data.shape[1])]
    designs = {
        "additive_multivariate": np.column_stack(base),
        "pairwise_interaction_multivariate": np.column_stack(base + pairwise),
        "quadratic_multivariate": np.column_stack(base + pairwise + squares),
    }
    if x_data.shape[1] >= 3:
        triple = np.prod(x_data, axis=1)
        designs["full_interaction_multivariate"] = np.column_stack(base + pairwise + [triple])
    return designs


def _multivariate_symbolic_basis(x_data, feature_names, feature_symbols):
    """Symbolic companions to the cheap multivariate design matrices."""
    ones = [(np.ones(len(x_data), dtype=np.float64), sp.Integer(1), [])]
    linear = [
        (x_data[:, idx], feature_symbols[name], [name])
        for idx, name in enumerate(feature_names)
    ]
    pairwise = [
        (
            x_data[:, i] * x_data[:, j],
            feature_symbols[feature_names[i]] * feature_symbols[feature_names[j]],
            [feature_names[i], feature_names[j]],
        )
        for i, j in _pairwise_indices(len(feature_names))
    ]
    squares = [
        (x_data[:, idx] ** 2, feature_symbols[name] ** 2, [name])
        for idx, name in enumerate(feature_names)
    ]
    families = {
        "additive_multivariate": ones + linear,
        "pairwise_interaction_multivariate": ones + linear + pairwise,
        "quadratic_multivariate": ones + linear + pairwise + squares,
    }
    if len(feature_names) >= 3:
        families["full_interaction_multivariate"] = (
            ones + linear + pairwise + [
                (
                    np.prod(x_data, axis=1),
                    sp.prod(feature_symbols[name] for name in feature_names),
                    list(feature_names),
                )
            ]
        )
    return families


def _family_routing_analysis(x_data, y_data, feature_names):
    """Cheap classifier used to decide whether symbolic regression is promising."""
    scores = []
    for idx, name in enumerate(feature_names):
        x_col = x_data[:, idx]
        _record_family_score(
            scores,
            "linear",
            np.polyval(np.polyfit(x_col, y_data, 1), x_col),
            y_data,
            f"Low residual affine fit on feature {name}.",
            features=[name],
        )
        degree = min(3, max(1, len(x_col) - 1))
        _record_family_score(
            scores,
            f"polynomial_degree_{degree}",
            np.polyval(np.polyfit(x_col, y_data, degree), x_col),
            y_data,
            f"Low-degree polynomial fit on feature {name} matches the data shape.",
            features=[name],
        )

        if np.all(x_col > 0):
            coeffs = np.polyfit(np.log(x_col), y_data, 1)
            _record_family_score(
                scores,
                "logarithmic",
                coeffs[0] * np.log(x_col) + coeffs[1],
                y_data,
                f"Linear fit in log({name}) space is competitive.",
                features=[name],
            )
        if np.all(y_data > 0):
            coeffs = np.polyfit(x_col, np.log(y_data), 1)
            _record_family_score(
                scores,
                "exponential",
                float(np.exp(coeffs[1])) * np.exp(coeffs[0] * x_col),
                y_data,
                f"Linear fit in log(y) vs {name} space is competitive.",
                features=[name],
            )
        if np.all(x_col > 0) and np.all(y_data > 0):
            coeffs = np.polyfit(np.log(x_col), np.log(y_data), 1)
            _record_family_score(
                scores,
                "power_law",
                float(np.exp(coeffs[1])) * np.power(x_col, coeffs[0]),
                y_data,
                f"Log-log fit on feature {name} is competitive.",
                features=[name],
            )

    if x_data.shape[1] >= 2:
        for family, design in _multivariate_designs(x_data).items():
            coeffs, *_ = np.linalg.lstsq(design, y_data, rcond=None)
            rationale = {
                "additive_multivariate": "Additive multivariate model explains most variance.",
                "pairwise_interaction_multivariate": "Pairwise interactions materially improve the fit.",
                "quadratic_multivariate": "Low-order polynomial terms explain the multivariate structure.",
                "full_interaction_multivariate": "A full three-way interaction is competitive on the observed samples.",
            }.get(family, "A multivariate linearized family is competitive.")
            _record_family_score(
                scores,
                family,
                design @ coeffs,
                y_data,
                rationale,
                features=feature_names,
            )

        if np.all(y_data > 0):
            design = np.column_stack(
                [np.ones(len(y_data), dtype=np.float64)] + [x_data[:, idx] for idx in range(x_data.shape[1])]
            )
            coeffs, *_ = np.linalg.lstsq(design, np.log(y_data), rcond=None)
            linear_part = sum(coeff * x_data[:, idx] for idx, coeff in enumerate(coeffs[1:]))
            _record_family_score(
                scores,
                "separable_exponential",
                float(np.exp(coeffs[0])) * np.exp(linear_part),
                y_data,
                "Log-linear fit suggests separable exponential structure.",
                features=feature_names,
            )
        if np.all(x_data > 0) and np.all(y_data > 0):
            design = np.column_stack(
                [np.ones(len(y_data), dtype=np.float64)] + [np.log(x_data[:, idx]) for idx in range(x_data.shape[1])]
            )
            coeffs, *_ = np.linalg.lstsq(design, np.log(y_data), rcond=None)
            pred = float(np.exp(coeffs[0])) * np.prod(
                [np.power(x_data[:, idx], coeffs[idx + 1]) for idx in range(x_data.shape[1])],
                axis=0,
            )
            _record_family_score(
                scores,
                "separable_power_law",
                pred,
                y_data,
                "Log-log fit suggests separable multiplicative structure.",
                features=feature_names,
            )

    return sorted(scores, key=lambda item: (item["normalized_rmse"], item["mse"]))


def analyze_dataset(x_data, y_data, feature_names=None):
    """Classify whether symbolic regression is worth attempting on a dataset."""
    x_data, y_data, feature_names = _coerce_regression_inputs(
        x_data,
        y_data,
        feature_names=feature_names,
    )
    scores = _family_routing_analysis(x_data, y_data, feature_names)
    best_family = scores[0] if scores else None
    periodic_score = _periodicity_score(y_data)
    piecewise_score = _piecewise_score(y_data)
    feature_importance = _feature_importance(x_data, y_data, feature_names)

    flags = []
    if len(y_data) < max(8, 6 * x_data.shape[1]):
        flags.append("too_few_points")
    if x_data.shape[1] > MAX_SEARCH_FEATURES:
        flags.append("too_many_features")
    if periodic_score > 0.55:
        flags.append("looks_periodic")
    if piecewise_score > 0.65:
        flags.append("appears_piecewise")
    if best_family and best_family["normalized_rmse"] < 0.03:
        if best_family["family"] == "exponential":
            flags.append("looks_exponential")
        elif best_family["family"] == "power_law":
            flags.append("looks_power_law")
        elif best_family["family"].startswith("separable_"):
            flags.append("looks_separable_multivariate")
    if best_family and best_family["normalized_rmse"] > 0.25 and periodic_score < 0.45:
        flags.append("probably_not_symbolic")

    should_attempt = (
        x_data.shape[1] <= MAX_SEARCH_FEATURES
        and len(y_data) >= max(6, 3 * x_data.shape[1])
        and "probably_not_symbolic" not in flags
    )

    if "too_many_features" in flags:
        recommended_mode = "instant"
    elif "too_few_points" in flags:
        recommended_mode = "instant"
    elif "looks_periodic" in flags or "appears_piecewise" in flags:
        recommended_mode = "balanced"
    elif best_family and best_family["normalized_rmse"] < 0.03:
        recommended_mode = "instant"
    elif x_data.shape[1] >= 2:
        recommended_mode = "balanced"
    else:
        recommended_mode = "deep"

    units_assumptions = [
        "No explicit unit inference is performed; input values are treated as dimensionless numbers.",
        "Any log/power-law hints assume strictly positive values on the transformed axes.",
    ]

    return {
        "sample_count": int(len(y_data)),
        "n_features": int(x_data.shape[1]),
        "feature_names": list(feature_names),
        "supported": bool(x_data.shape[1] <= MAX_SEARCH_FEATURES),
        "should_attempt_regression": bool(should_attempt),
        "recommended_mode": recommended_mode,
        "flags": flags,
        "likely_families": scores[:3],
        "periodicity_score": float(periodic_score),
        "piecewise_score": float(piecewise_score),
        "feature_importance": feature_importance,
        "units_assumptions": units_assumptions,
    }


def _predict_candidate(candidate, x_data, feature_names):
    """Evaluate any candidate formula back on the observed samples."""
    if candidate.get("leaf_types"):
        return evaluate_tree(
            candidate["leaf_types"],
            x_data,
            candidate.get("constants", []),
            feature_names=feature_names,
        )

    expression = candidate.get("expression")
    if expression is None:
        return np.full(len(x_data), np.nan, dtype=np.float64)

    expr = sp.sympify(expression)
    if not expr.free_symbols:
        return np.full(len(x_data), float(expr), dtype=np.float64)
    free_symbols = {sym.name: sym for sym in expr.free_symbols}
    ordered_names = [name for name in feature_names if name in free_symbols]
    fn = sp.lambdify([free_symbols[name] for name in ordered_names], expr, "numpy")
    args = [x_data[:, feature_names.index(name)] for name in ordered_names]
    pred = fn(*args)
    if np.isscalar(pred):
        return np.full(len(x_data), float(pred), dtype=np.float64)
    return np.asarray(pred, dtype=np.float64)


def _tree_candidate(leaf_types, mse, constants, depth, feature_names, feature_symbols,
                    y_data, tolerance):
    """Convert one EML tree into the common candidate/result shape."""
    try:
        sym = tree_to_sympy(leaf_types, constants or None, feature_symbols=feature_symbols)
    except Exception:
        sym = None
    expression = sym if sym is not None else _eml_str(leaf_types, constants)
    used_features = _used_features_from_leaf_types(leaf_types, feature_names)
    return _build_result(
        expression,
        mse,
        depth,
        leaf_types=leaf_types,
        constants=constants,
        eml_expression=_eml_str(leaf_types, constants),
        strategy="eml_tree",
        feature_names=feature_names,
        used_features=used_features,
        family="eml_tree",
        rationale=f"Best EML tree encountered at depth {depth}.",
        quality=_quality_from_metrics(mse, y_data, tolerance),
    )


def _residual_summary(pred, y_data):
    """Compact residual diagnostics for LLM-facing output."""
    mask = np.isfinite(pred)
    valid = int(mask.sum())
    if valid == 0:
        return {
            "valid_fraction": 0.0,
            "mae": float("inf"),
            "max_abs_error": float("inf"),
            "median_abs_error": float("inf"),
        }
    residuals = pred[mask] - y_data[mask]
    abs_resid = np.abs(residuals)
    return {
        "valid_fraction": float(valid / len(y_data)),
        "mae": float(np.mean(abs_resid)),
        "max_abs_error": float(np.max(abs_resid)),
        "median_abs_error": float(np.median(abs_resid)),
    }


def _constant_stability(best, x_data, y_data, feature_names):
    """Probe whether fitted constants are brittle to small perturbations."""
    if not best.get("leaf_types") or not best.get("constants"):
        return None
    constants = np.asarray(best["constants"], dtype=np.float64)
    deltas = np.array([0.99, 1.01], dtype=np.float64)
    ratios = []
    base_mse = max(best["mse"], 1e-12)
    for delta in deltas:
        trial = constants * delta
        pred = evaluate_tree(best["leaf_types"], x_data, trial, feature_names=feature_names)
        mse = _score_prediction(pred, y_data)
        ratios.append(float(mse / base_mse))
    return {
        "perturbation": 0.01,
        "mse_ratio_min": float(min(ratios)),
        "mse_ratio_max": float(max(ratios)),
    }


def _describe_regions(x_data, idxs, feature_names):
    """Summarise disagreement hotspots as concrete data-collection hints."""
    if len(idxs) == 0:
        return []
    descriptions = []
    if x_data.shape[1] == 1:
        for idx in idxs:
            descriptions.append(f"{feature_names[0]}~={x_data[idx, 0]:.3g}")
    else:
        for idx in idxs:
            parts = [f"{name}~={x_data[idx, col]:.3g}" for col, name in enumerate(feature_names)]
            descriptions.append(", ".join(parts))
    return descriptions


def _recommended_followup(best, candidates, x_data, feature_names):
    """Suggest the next measurement that would separate close alternatives."""
    if len(candidates) < 2:
        used = ", ".join(best.get("used_features") or feature_names)
        return f"Collect a few more points spanning the edges of the {used} range."
    best_pred = _predict_candidate(candidates[0], x_data, feature_names)
    alt_pred = _predict_candidate(candidates[1], x_data, feature_names)
    diff = np.abs(best_pred - alt_pred)
    mask = np.isfinite(diff)
    if not mask.any():
        return "Collect additional points in regions not covered by the current sample."
    ranked = np.argsort(diff[mask])[-3:]
    source_idx = np.nonzero(mask)[0][ranked]
    regions = _describe_regions(x_data, source_idx[::-1], feature_names)
    if not regions:
        return "Collect additional points in regions not covered by the current sample."
    return (
        "Collect extra data where the top two candidates disagree most: "
        + "; ".join(regions)
        + "."
    )


def _verification_summary(best, x_data, y_data, feature_names, candidates, analysis, tolerance):
    """Post-search checks used to separate verified from shaky fits."""
    pred = _predict_candidate(best, x_data, feature_names)
    residuals = _residual_summary(pred, y_data)
    holdout_mask = (np.arange(len(y_data)) % 5) == 0
    holdout_pred = pred[holdout_mask]
    holdout_y = y_data[holdout_mask]
    holdout_mse = _score_prediction(holdout_pred, holdout_y, min_valid_fraction=1.0)
    simpler_alt = None
    for candidate in candidates[1:]:
        if candidate["complexity"] <= best["complexity"]:
            simpler_alt = candidate
            break
    simpler_gap = None
    if simpler_alt is not None and np.isfinite(best["mse"]):
        simpler_gap = float(simpler_alt["mse"] / max(best["mse"], 1e-12))

    failure_modes = []
    if "too_few_points" in analysis["flags"]:
        failure_modes.append("insufficient_data")
    if "appears_piecewise" in analysis["flags"]:
        failure_modes.append("data_appears_piecewise")
    if "looks_periodic" in analysis["flags"] and best["quality"] != "exact":
        failure_modes.append("periodic_structure_detected_but_grammar_insufficient")
    if residuals["max_abs_error"] > 10 * max(residuals["median_abs_error"], 1e-12):
        failure_modes.append("fit_dominated_by_one_outlier")
    stability = _constant_stability(best, x_data, y_data, feature_names)
    if stability and stability["mse_ratio_max"] > 5.0:
        failure_modes.append("constant_optimization_unstable")
    if simpler_gap is not None and simpler_gap < 1.2:
        failure_modes.append("multiple_competing_candidates")
    if "probably_not_symbolic" in analysis["flags"] and best["quality"] == "rough":
        failure_modes.append("probably_not_symbolic")

    if best["quality"] == "exact" and not failure_modes:
        status = "verified"
    elif best["quality"] in {"exact", "approximate"}:
        status = "plausible"
    else:
        status = "weak"

    return {
        "status": status,
        "holdout_mse": float(holdout_mse),
        "holdout_normalized_rmse": _normalized_rmse(holdout_mse, holdout_y),
        "residual_summary": residuals,
        "simpler_alternative_gap": simpler_gap,
        "constant_stability": stability,
        "failure_modes": failure_modes,
    }


def _confidence_summary(best, analysis, verification):
    """Turn numeric diagnostics into a compact confidence object."""
    reasons = []
    score = 0.85 if best["quality"] == "exact" else 0.62 if best["quality"] == "approximate" else 0.32
    if best.get("strategy") == "prepass":
        score += 0.05
        reasons.append("A simple closed-form family fit matched before tree search.")
    if analysis.get("likely_families"):
        top_family = analysis["likely_families"][0]
        if top_family["normalized_rmse"] < 0.05:
            score += 0.05
            reasons.append(f"Cheap routing heuristics also favored {top_family['family']}.")
    for failure in verification.get("failure_modes", []):
        score -= 0.08
    holdout_nrmse = verification.get("holdout_normalized_rmse", float("inf"))
    if np.isfinite(holdout_nrmse) and holdout_nrmse < 0.10:
        reasons.append("Post-fit split residuals remain small.")
    elif np.isfinite(holdout_nrmse):
        score -= 0.05
        reasons.append("Post-fit split residuals are noticeably larger.")
    score = float(np.clip(score, 0.02, 0.99))
    label = "high" if score >= 0.80 else "medium" if score >= 0.55 else "low"
    return {
        "score": score,
        "label": label,
        "reasons": reasons,
    }


def _guidance_summary(best, analysis, verification, candidates, x_data, feature_names):
    """LLM-oriented scaffold for small-model tool use."""
    why_best = best.get("rationale") or f"Lowest observed error among {best.get('strategy') or 'search'} candidates."
    if verification["failure_modes"]:
        caveat = "Watch for: " + ", ".join(verification["failure_modes"]) + "."
    else:
        caveat = "No major failure signals were detected on the observed samples."
    followup = _recommended_followup(best, candidates, x_data, feature_names)
    conclusion = f"Best current formula: y = {best.get('expression') or 'unknown'}."
    return {
        "why_best": why_best,
        "when_not_to_trust": caveat,
        "follow_up_experiment": followup,
        "user_conclusion": conclusion,
    }


def _finalize_result(best, x_data, y_data, feature_names, analysis, candidate_board, tolerance,
                     budget=None):
    """Attach diagnostics, confidence, and candidates to the best fit."""
    best = dict(best)
    best["analysis"] = analysis
    candidates = _ranked_candidates(candidate_board, limit=5)
    if best.get("expression") is not None:
        best_sig = _candidate_signature(best)
        if all(_candidate_signature(candidate) != best_sig for candidate in candidates):
            _register_candidate(candidate_board, best)
            candidates = _ranked_candidates(candidate_board, limit=5)
    best["candidates"] = candidates
    best["quality"] = _quality_from_metrics(best["mse"], y_data, tolerance)
    verification = _verification_summary(best, x_data, y_data, feature_names, candidates or [best], analysis, tolerance)
    best["verification"] = verification
    best["failure_modes"] = verification["failure_modes"]
    best["confidence"] = _confidence_summary(best, analysis, verification)
    best["guidance"] = _guidance_summary(best, analysis, verification, candidates or [best], x_data, feature_names)
    if budget is not None:
        best["requested_mode"] = budget["requested_mode"]
        best["effective_mode"] = budget["effective_mode"]
        best["search_budget"] = {
            "max_depth": budget["max_depth"],
            "max_random": budget["max_random"],
        }
    return best


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
        if not _contains_feature(pair) and pair.count(C) > 1:
            return True

    if len(cfg) >= 4:
        for i in range(0, len(cfg), 4):
            group = cfg[i:i + 4]
            if not _contains_feature(group) and C in group:
                return True
    return False


def _sample_one_config(n_leaves, constant_count, rng, feature_names):
    """Sample one leaf configuration with a target constant count."""
    if n_leaves < 1:
        raise ValueError("n_leaves must be positive")

    max_constants = max(0, n_leaves - 1)
    constant_count = int(np.clip(constant_count, 0, max_constants))
    cfg = [None] * n_leaves

    if constant_count:
        c_positions = rng.choice(n_leaves, size=constant_count, replace=False)
        for idx in np.atleast_1d(c_positions):
            cfg[int(idx)] = C

    non_const_positions = [i for i, value in enumerate(cfg) if value is None]
    feature_position = int(rng.choice(non_const_positions))
    cfg[feature_position] = str(rng.choice(feature_names))

    remaining = [i for i in non_const_positions if i != feature_position]
    if remaining:
        choices = rng.choice(list(feature_names) + [ONE, ZERO], size=len(remaining), replace=True)
        for idx, choice in zip(remaining, choices):
            cfg[idx] = str(choice)
    return tuple(cfg)


def _sample_configs(n_leaves, max_random, rng, feature_names):
    """Stratified random sampling biased toward 1-2 constant trees."""
    if max_random <= 0:
        return []

    sampled = {}
    bucket_counts = {bucket: 0 for bucket in SAMPLE_BUCKET_WEIGHTS}
    bucket_targets = {
        bucket: max(1, int(round(max_random * weight)))
        for bucket, weight in SAMPLE_BUCKET_WEIGHTS.items()
    }

    # Keep the total close to max_random after rounding.
    overflow = sum(bucket_targets.values()) - max_random
    if overflow > 0:
        for bucket in sorted(bucket_targets, reverse=True):
            if overflow <= 0:
                break
            reducible = min(overflow, bucket_targets[bucket] - 1)
            bucket_targets[bucket] -= reducible
            overflow -= reducible

    for bucket, target in bucket_targets.items():
        attempts = 0
        while bucket_counts[bucket] < target:
            if attempts > target * 20:
                break
            attempts += 1
            if bucket == 4:
                constant_count = int(rng.integers(4, max(5, n_leaves)))
            else:
                constant_count = bucket
            cfg = _sample_one_config(n_leaves, constant_count, rng, feature_names)
            if cfg not in sampled:
                sampled[cfg] = None
                bucket_counts[min(cfg.count(C), 4)] += 1

    while len(sampled) < max_random:
        constant_count = int(rng.integers(0, max(1, n_leaves)))
        cfg = _sample_one_config(n_leaves, constant_count, rng, feature_names)
        if cfg not in sampled:
            sampled[cfg] = None
            bucket_counts[min(cfg.count(C), 4)] += 1

    return list(sampled)


def _init_screen_worker(x_data, y_data):
    """Load immutable arrays into each worker once."""
    global _WORKER_X_DATA, _WORKER_Y_DATA
    _WORKER_X_DATA = x_data
    _WORKER_Y_DATA = y_data


# ── Tree evaluation ─────────────────────────────────────────────────

def evaluate_tree(leaf_types, x_data, constants=None, feature_names=None):
    """Evaluate a full binary EML tree bottom-up. Pure numpy."""
    x_matrix, resolved_names = _coerce_feature_matrix(
        x_data,
        feature_names=feature_names,
        preserve_scalar_name=True,
    )
    feature_lookup = {
        name: x_matrix[:, idx]
        for idx, name in enumerate(resolved_names)
    }
    n_samples = x_matrix.shape[0]
    c_idx = 0
    leaves = []
    for lt in leaf_types:
        if _is_feature_token(lt):
            if lt not in feature_lookup:
                raise ValueError(f"Unknown feature leaf '{lt}' for feature names {resolved_names}")
            leaves.append(feature_lookup[lt])
        elif lt == ONE:
            leaves.append(np.ones(n_samples, dtype=np.float64))
        elif lt == ZERO:
            leaves.append(np.zeros(n_samples, dtype=np.float64))
        else:  # C
            leaves.append(np.full(n_samples, float(constants[c_idx]), dtype=np.float64))
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
                       restarts=3, quick=False, rng=None, use_basinhopping=None,
                       feature_names=None):
    """Fit trainable constants via Nelder-Mead.

    ~7ms per call (quick), ~15ms (full). Near-instant for 1-4 params.
    """
    if n_constants <= 0:
        pred = evaluate_tree(leaf_types, x_data, [], feature_names=feature_names)
        return _score_prediction(pred, y_data), []

    if rng is None:
        rng = np.random.default_rng()

    if quick:
        restarts = 1
        maxiter = max(200, 100 * n_constants)
        use_basinhopping = False
    else:
        maxiter = max(2000, 500 * n_constants)
        if use_basinhopping is None:
            use_basinhopping = n_constants >= 3
        if use_basinhopping:
            maxiter = max(400, 200 * n_constants)
    base_scales = [0.5, 2.0, 10.0]
    minimizer_kwargs = {
        "method": "Nelder-Mead",
        "options": {"maxiter": maxiter, "xatol": 1e-12, "fatol": 1e-12},
    }

    def objective(c):
        pred = evaluate_tree(leaf_types, x_data, c, feature_names=feature_names)
        mse = _score_prediction(pred, y_data)
        return mse if np.isfinite(mse) else 1e20

    best_mse, best_c = float("inf"), None
    effective_restarts = 1 if use_basinhopping else restarts
    for r in range(effective_restarts):
        if r < len(base_scales):
            scale = base_scales[r]
        else:
            scale = 10 ** rng.uniform(-1.0, 1.0)
        c0 = rng.normal(size=n_constants) * scale
        try:
            if use_basinhopping:
                res = basinhopping(
                    objective,
                    c0,
                    minimizer_kwargs=minimizer_kwargs,
                    niter=max(4, 2 * n_constants),
                    T=1.0,
                    stepsize=2.0,
                    seed=int(rng.integers(0, 2**32 - 1)),
                )
            else:
                res = minimize(objective, c0, **minimizer_kwargs)

            fun = float(getattr(res, "fun", float("inf")))
            x_opt = getattr(res, "x", None)
            if x_opt is not None and fun < best_mse:
                best_mse = fun
                best_c = np.asarray(x_opt, dtype=float).tolist()
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


def _maybe_prepass_result(pred, expr, y_data, tolerance, feature_names, used_features,
                          candidate_board=None, family=None, rationale=None):
    """Wrap an exact or near-exact pre-pass fit into the standard result shape."""
    mse = _score_prediction(pred, y_data, min_valid_fraction=1.0)
    candidate = _build_result(
        expr,
        mse,
        depth=0,
        strategy="prepass",
        feature_names=feature_names,
        used_features=used_features,
        family=family,
        rationale=rationale,
        quality=None,
    )
    if candidate_board is not None:
        candidate["quality"] = _quality_from_metrics(mse, y_data, tolerance)
        _register_candidate(candidate_board, candidate)
    if mse < tolerance:
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
        return _build_result(
            expr,
            mse,
            depth=0,
            strategy="prepass",
            feature_names=feature_names,
            used_features=used_features,
            family=family,
            rationale=rationale,
        )
    return None


def _fit_linear_basis(basis_terms, y_data, tolerance, feature_names, nice_tol,
                      candidate_board=None, family=None, rationale=None):
    """Fit a linear combination of fixed basis functions with least squares."""
    if not basis_terms:
        return None
    design = np.column_stack([column for column, _, _ in basis_terms])
    if not np.all(np.isfinite(design)):
        return None

    coeffs, *_ = np.linalg.lstsq(design, y_data, rcond=None)
    pred = design @ coeffs
    expr = sp.Integer(0)
    used = set()
    for coeff, (_, term_expr, term_used) in zip(coeffs, basis_terms):
        if abs(coeff) < nice_tol:
            continue
        expr += _nice(float(coeff), tol=nice_tol) * term_expr
        used.update(term_used)
    return _maybe_prepass_result(
        pred,
        sp.expand(expr),
        y_data,
        tolerance,
        feature_names,
        [name for name in feature_names if name in used],
        candidate_board=candidate_board,
        family=family,
        rationale=rationale,
    )


def _fit_standard_forms_1d(x_data, y_data, tolerance, x_sym, feature_name, feature_names,
                           candidate_board=None):
    """Cheap exact-fit pre-pass for common one-variable function families."""
    nice_tol = max(tolerance, 1e-5)

    def _maybe_result(pred, expr, family, rationale):
        used = [feature_name] if x_sym in getattr(expr, "free_symbols", set()) else []
        return _maybe_prepass_result(
            pred,
            expr,
            y_data,
            tolerance,
            feature_names,
            used,
            candidate_board=candidate_board,
            family=family,
            rationale=rationale,
        )

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
        result = _maybe_result(
            pred,
            sp.expand(expr),
            family=f"polynomial_degree_{degree}",
            rationale=f"Degree-{degree} polynomial fit matched the sampled data.",
        )
        if result is not None:
            return result

    if np.all(x_data > 0) and np.all(y_data > 0):
        slope, intercept = np.polyfit(np.log(x_data), np.log(y_data), 1)
        scale = float(np.exp(intercept))
        pred = scale * np.power(x_data, slope)
        expr = _nice(scale, tol=nice_tol) * x_sym ** _nice(slope, tol=nice_tol)
        result = _maybe_result(
            pred,
            expr,
            family="power_law",
            rationale="A log-log linearization strongly favors a power-law relationship.",
        )
        if result is not None:
            return result

    if np.all(y_data > 0):
        slope, intercept = np.polyfit(x_data, np.log(y_data), 1)
        scale = float(np.exp(intercept))
        pred = scale * np.exp(slope * x_data)
        expr = _nice(scale, tol=nice_tol) * sp.exp(_nice(slope, tol=nice_tol) * x_sym)
        result = _maybe_result(
            pred,
            expr,
            family="exponential",
            rationale="A log-linear transform yields a strong exponential fit.",
        )
        if result is not None:
            return result

    if np.all(x_data > 0):
        slope, intercept = np.polyfit(np.log(x_data), y_data, 1)
        pred = slope * np.log(x_data) + intercept
        expr = _nice(slope, tol=nice_tol) * sp.log(x_sym) + _nice(intercept, tol=nice_tol)
        result = _maybe_result(
            pred,
            expr,
            family="logarithmic",
            rationale="A linear fit in log(x) space explains the samples well.",
        )
        if result is not None:
            return result

    return None


def _fit_standard_forms(x_data, y_data, tolerance, feature_names, feature_symbols,
                        candidate_board=None):
    """Hybrid pre-pass for exact common forms before EML tree search."""
    if x_data.shape[1] == 1:
        name = feature_names[0]
        return _fit_standard_forms_1d(
            x_data[:, 0],
            y_data,
            tolerance,
            feature_symbols[name],
            name,
            feature_names,
            candidate_board=candidate_board,
        )

    for idx, name in enumerate(feature_names):
        result = _fit_standard_forms_1d(
            x_data[:, idx],
            y_data,
            tolerance,
            feature_symbols[name],
            name,
            feature_names,
            candidate_board=candidate_board,
        )
        if result is not None:
            return result

    nice_tol = max(tolerance, 1e-5)

    multivariate_families = _multivariate_symbolic_basis(
        x_data,
        feature_names,
        feature_symbols,
    )
    for family, basis_terms in multivariate_families.items():
        rationale = {
            "additive_multivariate": "An additive multivariate basis already explains the observed data.",
            "pairwise_interaction_multivariate": "Pairwise multivariate interactions explain the observed data.",
            "quadratic_multivariate": "A low-order multivariate polynomial basis explains the observed data.",
            "full_interaction_multivariate": "A three-way interaction term materially improves the multivariate fit.",
        }.get(family, "A multivariate basis already explains the observed data.")
        result = _fit_linear_basis(
            basis_terms,
            y_data,
            tolerance,
            feature_names,
            nice_tol,
            candidate_board=candidate_board,
            family=family,
            rationale=rationale,
        )
        if result is not None:
            return result

    if np.all(y_data > 0):
        design = np.column_stack(
            [np.ones(len(y_data), dtype=np.float64)] + [x_data[:, idx] for idx in range(x_data.shape[1])]
        )
        coeffs, *_ = np.linalg.lstsq(design, np.log(y_data), rcond=None)
        scale = float(np.exp(coeffs[0]))
        linear_part = sum(coeffs[idx + 1] * x_data[:, idx] for idx in range(x_data.shape[1]))
        pred = scale * np.exp(linear_part)
        expr = _nice(scale, tol=nice_tol) * sp.exp(
            sum(
                _nice(float(coeffs[idx + 1]), tol=nice_tol) * feature_symbols[name]
                for idx, name in enumerate(feature_names)
            )
        )
        result = _maybe_prepass_result(
            pred,
            expr,
            y_data,
            tolerance,
            feature_names,
            feature_names,
            candidate_board=candidate_board,
            family="separable_exponential",
            rationale="A log-linear fit across both features suggests separable exponential structure.",
        )
        if result is not None:
            return result

    if np.all(x_data > 0) and np.all(y_data > 0):
        design = np.column_stack(
            [np.ones(len(y_data), dtype=np.float64)] + [np.log(x_data[:, idx]) for idx in range(x_data.shape[1])]
        )
        coeffs, *_ = np.linalg.lstsq(design, np.log(y_data), rcond=None)
        scale = float(np.exp(coeffs[0]))
        pred = scale * np.prod(
            [np.power(x_data[:, idx], coeffs[idx + 1]) for idx in range(x_data.shape[1])],
            axis=0,
        )
        expr = _nice(scale, tol=nice_tol) * sp.prod(
            feature_symbols[name] ** _nice(float(coeffs[idx + 1]), tol=nice_tol)
            for idx, name in enumerate(feature_names)
        )
        result = _maybe_prepass_result(
            pred,
            expr,
            y_data,
            tolerance,
            feature_names,
            feature_names,
            candidate_board=candidate_board,
            family="separable_power_law",
            rationale="A log-log fit across both features suggests separable multiplicative structure.",
        )
        if result is not None:
            return result

        coeffs, *_ = np.linalg.lstsq(design, y_data, rcond=None)
        pred = coeffs[0] + sum(coeffs[idx + 1] * np.log(x_data[:, idx]) for idx in range(x_data.shape[1]))
        expr = _nice(float(coeffs[0]), tol=nice_tol) + sum(
            _nice(float(coeffs[idx + 1]), tol=nice_tol) * sp.log(feature_symbols[name])
            for idx, name in enumerate(feature_names)
        )
        result = _maybe_prepass_result(
            pred,
            expr,
            y_data,
            tolerance,
            feature_names,
            feature_names,
            candidate_board=candidate_board,
            family="separable_logarithmic",
            rationale="A linear combination of feature logs is competitive.",
        )
        if result is not None:
            return result

    return None


def tree_to_sympy(leaf_types, constants=None, x_symbol=None, feature_symbols=None):
    """Convert an EML tree to a simplified sympy expression."""
    feature_symbols = dict(feature_symbols or {})
    if x_symbol is not None:
        feature_symbols.setdefault(X, x_symbol)
    c_idx = 0
    leaves = []
    for lt in leaf_types:
        if _is_feature_token(lt):
            leaves.append(feature_symbols.get(lt, sp.Symbol(lt, real=True)))
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
        if _is_feature_token(lt):
            strs.append(str(lt))
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
    if len(args) == 4:
        lt, nc, seed, feature_names = args
        x_data, y_data = _WORKER_X_DATA, _WORKER_Y_DATA
    else:
        lt, x_data, y_data, nc, seed, feature_names = args
    mse, consts = optimize_constants(
        lt,
        x_data,
        y_data,
        nc,
        quick=True,
        rng=np.random.default_rng(seed),
        feature_names=feature_names,
    )
    return mse, lt, nc, consts


def symbolic_regression(x_data, y_data, max_depth=None, tolerance=1e-8,
                        verbose=True, max_random=None, workers=None, seed=None,
                        feature_names=None, mode=None):
    """Search for the simplest EML tree that fits the data.

    A cheap exact-form pre-pass runs before the EML search. If that misses,
    each depth level uses three phases:
      1) Instant eval of constant-free trees (pure numpy)
      2) Parallel quick-screen of trees with trainable constants
      3) Full optimisation of top candidates

    Args:
        x_data:     Input vector or feature matrix.
        y_data:     Target array.
        max_depth:  Optional explicit tree-depth override.
        tolerance:  Early-stop MSE threshold.
        verbose:    Print progress.
        max_random: Optional explicit sample count when exhaustive search is too large.
        workers:    Parallel workers (default: auto).
        seed:       Optional RNG seed for deterministic search/optimisation.
        mode:       Search preset: instant, balanced, deep, research, or auto.

    Returns:
        dict with expression, eml_expression, mse, depth, constants, leaf_types.
    """
    x_data, y_data, feature_names = _coerce_regression_inputs(
        x_data,
        y_data,
        feature_names=feature_names,
    )
    if x_data.shape[1] > MAX_SEARCH_FEATURES:
        raise ValueError(
            f"EML tree search currently supports at most {MAX_SEARCH_FEATURES} variables"
        )
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)
    rng = np.random.default_rng(seed)
    feature_symbols = _feature_symbol_map(x_data, feature_names)
    leaf_options = list(feature_names) + [ONE, ZERO, C]
    analysis = analyze_dataset(x_data, y_data, feature_names=feature_names)
    candidate_board = _empty_candidate_board()
    budget = _resolve_search_budget(mode, max_depth, max_random, analysis)
    max_depth = budget["max_depth"]
    max_random = budget["max_random"]

    best = _build_result(
        None,
        float("inf"),
        depth=None,
        strategy=None,
        feature_names=feature_names,
    )
    t0 = time.time()

    def _finish(result):
        final = _finalize_result(
            result,
            x_data,
            y_data,
            feature_names,
            analysis,
            candidate_board,
            tolerance,
            budget=budget,
        )
        if verbose:
            _report(final, time.time() - t0)
        return final

    def _update(lt, mse, consts, depth):
        if mse < best["mse"]:
            candidate = _tree_candidate(
                lt,
                mse,
                consts,
                depth,
                feature_names,
                feature_symbols,
                y_data,
                tolerance,
            )
            _register_candidate(candidate_board, candidate)
            best.update(candidate)
            if verbose and mse < 1.0:
                print(f"  new best  MSE={mse:.2e}  {best['expression']}")
            return mse < tolerance
        return False

    prepass = _fit_standard_forms(
        x_data,
        y_data,
        tolerance,
        feature_names,
        feature_symbols,
        candidate_board=candidate_board,
    )
    if prepass is not None:
        if verbose:
            print("[pre-pass] exact fit found")
        return _finish(prepass)

    for depth in range(1, max_depth + 1):
        n_leaves = 2 ** depth
        n_cfgs = len(leaf_options) ** n_leaves
        exhaustive = n_cfgs <= EXHAUSTIVE_LIMIT

        if exhaustive:
            all_cfgs = list(product(leaf_options, repeat=n_leaves))
        else:
            all_cfgs = _sample_configs(n_leaves, max_random, rng, feature_names)

        all_cfgs = [c for c in all_cfgs if _contains_feature(c)]

        if verbose:
            search_mode = "exhaustive" if exhaustive else f"sampling {max_random}"
            print(
                f"[depth {depth}] {n_leaves} leaves | {search_mode} | "
                f"{len(all_cfgs)} viable | budget={budget['effective_mode']}"
            )

        # Phase 1: constant-free trees (instant)
        no_const = [c for c in all_cfgs if C not in c]
        leaf_cache = {
            (name,): x_data[:, idx]
            for idx, name in enumerate(feature_names)
        }
        leaf_cache[(ONE,)] = np.ones(len(y_data), dtype=np.float64)
        leaf_cache[(ZERO,)] = np.zeros(len(y_data), dtype=np.float64)
        sample_idx = np.linspace(0, len(y_data) - 1, num=min(5, len(y_data)), dtype=int)
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
                return _finish(best)

        if verbose:
            print(f"  phase 1  ({len(unique_no_const):>5} unique trees, no constants): "
                  f"best MSE = {best['mse']:.2e}")

        # Phase 2: parallel quick screen
        has_const = [c for c in all_cfgs if C in c and not _should_prune_config(c)]
        task_specs = [
            (list(cfg), list(cfg).count(C), int(rng.integers(0, 2**32 - 1)), feature_names)
            for cfg in has_const
        ]
        candidates = []

        if len(task_specs) > 100 and workers > 1:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_screen_worker,
                initargs=(x_data, y_data),
            )
            early_stop = False
            try:
                for mse, lt, nc, consts in pool.map(_screen_one, task_specs, chunksize=64):
                    if consts is None:
                        continue
                    candidates.append((mse, lt, nc, consts))
                    if _update(lt, mse, consts, depth) and mse < tolerance:
                        early_stop = True
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
            finally:
                if not early_stop:
                    pool.shutdown()
            if early_stop:
                return _finish(best)
        else:
            for lt, nc, task_seed, task_feature_names in task_specs:
                mse, lt, nc, consts = _screen_one(
                    (lt, x_data, y_data, nc, task_seed, task_feature_names)
                )
                if consts is None:
                    continue
                candidates.append((mse, lt, nc, consts))
                if _update(lt, mse, consts, depth) and mse < tolerance:
                    return _finish(best)

        if verbose:
            print(f"  phase 2  ({len(has_const):>5} trees, screened):    "
                  f"best MSE = {best['mse']:.2e}")

        if best["mse"] < tolerance:
            return _finish(best)

        # Phase 3: full optimisation on top candidates
        candidates.sort(key=lambda c: c[0])
        for _, lt, nc, _ in candidates[:30]:
            mse, consts = optimize_constants(
                lt,
                x_data,
                y_data,
                nc,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
                use_basinhopping=False,
                feature_names=feature_names,
            )
            if consts is None:
                continue
            _register_candidate(
                candidate_board,
                _tree_candidate(
                    lt,
                    mse,
                    consts,
                    depth,
                    feature_names,
                    feature_symbols,
                    y_data,
                    tolerance,
                ),
            )
            if _update(lt, mse, consts, depth) and mse < tolerance:
                return _finish(best)

        if verbose:
            print(f"  phase 3  (top 30, full optimise):   "
                  f"best MSE = {best['mse']:.2e}")

    return _finish(best)


def _report(b, elapsed):
    print(f"\n  {'='*50}")
    print(f"  Formula : {b.get('expression') or 'none'}")
    print(f"  EML tree: {b.get('eml_expression') or 'n/a'}")
    print(f"  MSE     : {b['mse']:.2e}")
    print(f"  Quality : {b.get('quality') or 'n/a'}")
    if b.get("effective_mode"):
        print(f"  Mode    : {b.get('effective_mode')} (requested {b.get('requested_mode')})")
    depth = b.get("depth")
    depth_label = depth if depth is not None else "n/a"
    used = ", ".join(b.get("used_features") or []) or "n/a"
    print(f"  Features: {used}")
    confidence = b.get("confidence", {})
    if confidence:
        print(f"  Trust   : {confidence.get('label', 'n/a')} ({confidence.get('score', 0.0):.2f})")
    failures = ", ".join(b.get("failure_modes") or []) or "none"
    print(f"  Alerts  : {failures}")
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
                    help='JSON data file: {"x":[...],"y":[...]} or {"x":[[...],[...]],"y":[...]}')
    ap.add_argument("--func", type=str,
                    help='Generate 1D data from expression, e.g. "sin(x)"')
    ap.add_argument("--range", nargs=2, type=float, default=[0.1, 5.0],
                    metavar=("LO", "HI"),
                    help="x range for --func (default: 0.1 5.0)")
    ap.add_argument("--points", type=int, default=200,
                    help="Number of data points (default: 200)")
    ap.add_argument("--max-depth", type=int,
                    help="Optional explicit maximum tree depth override")
    ap.add_argument("--mode", type=str, default="auto",
                    choices=["auto", "instant", "balanced", "deep", "research"],
                    help="Search budget preset (default: auto)")
    ap.add_argument("--tolerance", type=float, default=1e-8,
                    help="MSE tolerance for early stop (default: 1e-8)")
    ap.add_argument("--seed", type=int,
                    help="RNG seed for deterministic search")
    ap.add_argument("--output-json", action="store_true",
                    help="Output result as JSON")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress progress output")
    args = ap.parse_args()

    if args.demo:
        _run_demo()
        return

    feature_names = None
    if args.json:
        with open(args.json) as f:
            d = json.load(f)
        xd, yd = np.array(d["x"], float), np.array(d["y"], float)
        feature_names = d.get("feature_names")
        args.mode = d.get("mode", args.mode)
    elif args.func:
        xd = np.linspace(args.range[0], args.range[1], args.points)
        yd = sp.lambdify(sp.Symbol("x"), sp.sympify(args.func), "numpy")(xd)
    elif not sys.stdin.isatty():
        d = json.load(sys.stdin)
        xd, yd = np.array(d["x"], float), np.array(d["y"], float)
        feature_names = d.get("feature_names")
        args.max_depth = d.get("max_depth", args.max_depth)
        args.mode = d.get("mode", args.mode)
        args.seed = d.get("seed", args.seed)
        args.output_json = True
        args.quiet = True
    else:
        ap.print_help()
        return

    r = symbolic_regression(xd, yd, max_depth=args.max_depth,
                            tolerance=args.tolerance, verbose=not args.quiet,
                            seed=args.seed, feature_names=feature_names,
                            mode=args.mode)
    if args.output_json:
        print(json.dumps(r, indent=2))


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
