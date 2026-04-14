#!/usr/bin/env python3
"""
EML Symbolic Regression -- MCP stdio server.

Uses the official MCP Python SDK (FastMCP) so framing/protocol is always
compatible with whatever version hermes-agent uses.

Configure in ~/.hermes/config.yaml:

    mcp_servers:
      eml:
        command: python
        args: ["/path/to/eml-symbolic-regression/hermes/mcp_server.py"]
        timeout: 120

Hermes auto-discovers the tool as mcp_eml_eml_symbolic_regression.

Requires: pip install mcp eml-symbolic-regression
"""

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    import sys
    print("ERROR: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

import json
import numpy as np

mcp = FastMCP(
    name="eml-symbolic-regression",
    version="0.1.0",
)


@mcp.tool()
def eml_symbolic_regression(
    x: list[float],
    y: list[float],
    max_depth: int = 3,
) -> str:
    """Discover the mathematical formula underlying numerical data.

    Given x and y arrays, finds the simplest closed-form expression y=f(x)
    using EML binary trees. Returns the formula in standard math notation.

    Args:
        x: Input x values (at least 20 points, avoid x=0).
        y: Output y values corresponding to each x.
        max_depth: Max tree depth 1-4. 3 covers most functions.
    """
    from eml import regress

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
    }, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
