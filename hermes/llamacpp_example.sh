#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Example: Running Hermes with EML tool via llama.cpp server
# ─────────────────────────────────────────────────────────────────
#
# 1. Start llama.cpp server with tool support
# 2. Send a request with the EML tool definition
# 3. When Hermes calls the tool, pipe args to eml-regress
#
# Prerequisites:
#   pip install eml-symbolic-regression
#   # or: pip install -e /path/to/eml-symbolic-regression
# ─────────────────────────────────────────────────────────────────

# The tool definition (load from hermes/tool_schema.json)
TOOL_DEF=$(cat hermes/tool_schema.json)

# Example: Hermes decides to call the tool with these arguments
TOOL_ARGS='{"x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], "y": [1.6487, 2.7183, 4.4817, 7.3891, 12.1825, 20.0855, 33.1155, 54.5982], "max_depth": 2}'

# Execute the tool
echo "$TOOL_ARGS" | python -m eml.engine

# Output:
# {
#   "expression": "exp(x)",
#   "eml_expression": "eml(x, 1)",
#   "depth": 1,
#   "mse": 1.13e-09,
#   ...
# }
