# EML Symbolic Regression Tool

You have access to `eml_symbolic_regression`, a tool that discovers mathematical formulas from numerical data.

## When to use it

- The user provides data points and asks "what's the formula?"
- The user describes a pattern and you need to identify the function
- You need to verify a mathematical relationship
- You need to convert measurements into a symbolic expression

## How to use it

1. Extract or generate numerical x and y values from the user's problem
2. Call the tool with at least 20 data points (more is better)
3. Avoid x=0 in the data (ln(0) is undefined)
4. Start with max_depth=2 for fast results, increase to 3 if needed

## Interpreting results

- `expression`: The formula in standard math notation
- `eml_expression`: The EML tree structure (for reference)
- `mse`: Fit quality. Below 1e-8 = essentially exact. Above 1e-2 = approximation.

## Example

User: "These values seem to follow a pattern: (1, 2.72), (2, 7.39), (3, 20.09)"

You should:
1. Call eml_symbolic_regression with x=[1,2,3] and y=[2.72,7.39,20.09]
2. The tool will return expression="exp(x)"
3. Report: "The data follows an exponential function: y = e^x"
