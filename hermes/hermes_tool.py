"""
EML Symbolic Regression -- Hermes Agent native tool module.

Drop this file into your hermes-agent/tools/ directory,
then make sure it gets imported during agent startup.

Requires: pip install eml-symbolic-regression
"""

import json
import numpy as np

# ── Schema (inner format -- registry wraps it in {"type":"function",...}) ──

EML_SCHEMA = {
    "name": "eml_symbolic_regression",
    "description": (
        "Discover the mathematical formula underlying numerical data. "
        "Given x and y arrays, finds the simplest closed-form expression y=f(x). "
        "Uses the EML operator eml(a,b)=exp(a)-ln(b) which can represent ALL "
        "elementary functions. Returns the formula in standard math notation. "
        "Use when: finding a formula from data points, identifying a function "
        "behind a pattern, verifying a mathematical relationship, or converting "
        "numerical data to symbolic form."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Input x values (at least 20 points, avoid x=0)",
            },
            "y": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Output y values corresponding to each x",
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "description": (
                    "Max tree depth 1-4. "
                    "1-2: basic ops (exp, ln, powers). "
                    "3: most elementary functions. "
                    "4: complex compositions (slower)."
                ),
            },
        },
        "required": ["x", "y"],
    },
}


# ── Handler ─────────────────────────────────────────────────────────

def _handle_eml_regression(args: dict, **kwargs) -> str:
    """Hermes tool handler. Receives parsed args, returns JSON string."""
    try:
        from eml import regress
    except ImportError:
        return json.dumps({
            "error": "eml-symbolic-regression not installed. "
                     "Run: pip install eml-symbolic-regression"
        })

    x = args.get("x")
    y = args.get("y")
    max_depth = args.get("max_depth", 3)

    if not x or not y:
        return json.dumps({"error": "Both 'x' and 'y' arrays are required."})
    if len(x) != len(y):
        return json.dumps({"error": f"x ({len(x)}) and y ({len(y)}) must be same length."})
    if len(x) < 3:
        return json.dumps({"error": "Need at least 3 data points (20+ recommended)."})

    try:
        result = regress(
            np.array(x, dtype=np.float64),
            np.array(y, dtype=np.float64),
            max_depth=int(max_depth),
            verbose=False,
        )
    except Exception as e:
        return json.dumps({"error": f"Regression failed: {str(e)}"})

    mse = result["mse"]
    quality = "exact" if mse < 1e-8 else "approximate" if mse < 1e-1 else "rough"

    return json.dumps({
        "expression": result["expression"],
        "eml_expression": result["eml_expression"],
        "depth": result["depth"],
        "mse": mse,
        "quality": quality,
        "constants": result["constants"],
    })


# ── Registration ────────────────────────────────────────────────────
# This runs at import time when Hermes loads the tools/ directory.

def register(registry):
    """Called by Hermes tool loader, or call manually after import."""
    registry.register(
        name="eml_symbolic_regression",
        toolset="math",
        schema=EML_SCHEMA,
        handler=_handle_eml_regression,
        emoji="🔬",
        max_result_size_chars=10_000,
    )


# Auto-register if the Hermes registry singleton is available
try:
    from tools.registry import registry
    register(registry)
except ImportError:
    # Not running inside hermes-agent -- that's fine.
    # Users can call register(their_registry) manually.
    pass
