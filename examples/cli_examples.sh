#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# CLI usage examples
# ─────────────────────────────────────────────────────────────────

# 1. Discover formula from a known function (generates data internally)
eml-regress --func "exp(x)" --range 0.1 5 --max-depth 2

# 2. Pipe JSON data (how an LLM calls the tool)
echo '{"x": [1,2,3,4,5], "y": [2.72,7.39,20.09,54.60,148.41]}' | eml-regress

# 3. From a JSON file
eml-regress --json measurements.json --max-depth 3

# 4. Get JSON output (for programmatic use)
echo '{"x": [1,2,3,4], "y": [1,0.5,0.333,0.25]}' | eml-regress --output-json

# 5. Quick test at depth 2 only
eml-regress --func "1/x" --range 0.1 5 --max-depth 2

# 6. Run the built-in demo
eml-regress --demo
