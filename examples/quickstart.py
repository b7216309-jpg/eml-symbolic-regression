"""
Quickstart: discover exp(x) from data points.
"""
from eml import regress
import numpy as np

x = np.linspace(0.1, 5.0, 200)
y = np.exp(x)

result = regress(x, y)

print(f"Formula:  {result['expression']}")
print(f"EML tree: {result['eml_expression']}")
print(f"MSE:      {result['mse']:.2e}")
