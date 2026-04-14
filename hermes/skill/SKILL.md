---
name: symbolic-regression
description: Discover mathematical formulas from numerical data using EML trees
version: "0.1.0"
tags:
  - math
  - science
  - data-analysis
  - symbolic-regression
requires_tools:
  - eml_symbolic_regression
---

# Symbolic Regression with EML

Use this skill when the user wants to find the mathematical formula behind numerical data.

## Workflow

1. **Collect data**: Extract or generate x and y values from the user's problem.
   - Need at least 20 data points for reliable results (50-200 is ideal).
   - Avoid x=0 (the EML operator uses ln, which is undefined at 0).
   - Spread points evenly across the range of interest.

2. **Call the tool**: Use `eml_symbolic_regression` with the data.
   - Start with `max_depth=2` for a fast pass (~2-5 seconds).
   - If the result has MSE > 0.01, retry with `max_depth=3` (~30 seconds).

3. **Interpret the result**:
   - `quality: "exact"` (MSE < 1e-8): The formula is essentially perfect.
   - `quality: "approximate"` (MSE < 0.1): Close but may need deeper search.
   - `quality: "rough"` (MSE >= 0.1): The function may need depth 4 or may not be elementary.

4. **Explain to the user**:
   - Report the formula in standard notation.
   - Mention the EML tree form if the user is interested in the theory.
   - If approximate, explain the limitation and suggest alternatives.

## What works best

- Exponential functions: exp(x), exp(-x), exp(2x)
- Logarithms: ln(x), log_b(x)
- Reciprocals and powers: 1/x, 1/x^2
- Compositions: exp(ln(x)), ln(exp(x)-1)
- Physics/biology/finance formulas (typically exp/ln based)

## What's harder

- Trigonometric functions (sin, cos) need depth 4+ for exact results
- Polynomials (x+1, x^2) are counterintuitively hard in EML
- For those, consider using the `execute_code` tool with numpy.polyfit instead

## Tips

- If the user has noisy data, suggest averaging or smoothing first
- For multi-variable data, this tool only handles y=f(x) -- explain the limitation
- The `eml_expression` in the output shows the raw tree, useful for teaching
