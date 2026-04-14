"""
EML Symbolic Regression -- Hermes Agent native tool module.

Integration requires TWO changes to hermes-agent:

1. Add this import to tools/model_tools.py in _discover_tools():
       "tools.eml_tool",

2. Add to tools/toolsets.py:
       - Add "eml_symbolic_regression" to _HERMES_CORE_TOOLS
       - Or create a toolset: "math": ToolsetDef("Mathematical tools",
             tools=["eml_symbolic_regression"])

Or skip both and use the plugin system instead (see README).

Requires: pip install eml-symbolic-regression
"""

import json
import numpy as np


# ── Availability check ──────────────────────────────────────────────

def _check_eml_available() -> bool:
    """Check if eml-symbolic-regression is installed."""
    try:
        from eml import regress  # noqa: F401
        return True
    except ImportError:
        return False


# ── Schema ──────────────────────────────────────────────────────────

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
    """Hermes tool handler. Receives parsed args dict, returns JSON string.

    kwargs may contain task_id, user_task from Hermes dispatch.
    """
    # Import helpers -- fall back to plain json if not inside hermes-agent
    try:
        from tools.registry import tool_error, tool_result
    except ImportError:
        tool_error = lambda msg, **kw: json.dumps({"error": msg, **kw})
        tool_result = lambda **kw: json.dumps(kw)

    try:
        from eml import regress
    except ImportError:
        return tool_error(
            "eml-symbolic-regression not installed. "
            "Run: pip install eml-symbolic-regression"
        )

    x = args.get("x")
    y = args.get("y")
    max_depth = args.get("max_depth", 3)

    if not x or not y:
        return tool_error("Both 'x' and 'y' arrays are required.")
    if len(x) != len(y):
        return tool_error(f"x ({len(x)}) and y ({len(y)}) must be same length.")
    if len(x) < 3:
        return tool_error("Need at least 3 data points (20+ recommended).")

    try:
        result = regress(
            np.array(x, dtype=np.float64),
            np.array(y, dtype=np.float64),
            max_depth=int(max_depth),
            verbose=False,
        )
    except Exception as e:
        return tool_error(f"Regression failed: {str(e)}")

    mse = result["mse"]
    quality = "exact" if mse < 1e-8 else "approximate" if mse < 1e-1 else "rough"

    return json.dumps({
        "expression": result["expression"],
        "eml_expression": result["eml_expression"],
        "depth": result["depth"],
        "strategy": result.get("strategy"),
        "mse": mse,
        "quality": quality,
        "constants": result["constants"],
    })


# ── Registration ────────────────────────────────────────────────────

def register(registry):
    """Register the EML tool with a Hermes ToolRegistry instance."""
    registry.register(
        name="eml_symbolic_regression",
        toolset="math",
        schema=EML_SCHEMA,
        handler=_handle_eml_regression,
        check_fn=_check_eml_available,
        requires_env=[],
        is_async=False,
        emoji="\U0001f52c",  # microscope
        max_result_size_chars=10_000,
    )


# Auto-register when imported inside hermes-agent
try:
    from tools.registry import registry
    register(registry)
except ImportError:
    # Not running inside hermes-agent -- that's fine.
    # Users can call register(their_registry) manually.
    pass
