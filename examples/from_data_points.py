"""
Discover a formula from raw measurement data.

Scenario: you measured voltage decay across a capacitor
and want to know the underlying law.
"""
from eml import regress
import numpy as np

# Simulated RC circuit: V(t) = V0 * exp(-t/RC)
# V0 = 5V, RC = 2s
t = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
              4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0])
V = np.array([4.76, 4.31, 3.89, 3.30, 3.03, 2.36, 1.84, 1.43, 1.12, 0.87,
              0.68, 0.53, 0.41, 0.32, 0.25, 0.19, 0.15, 0.12, 0.09, 0.06, 0.03])

result = regress(t, V, max_depth=2, verbose=True)

print(f"\nDiscovered: V(t) = {result['expression']}")
print(f"MSE: {result['mse']:.2e}")
print(f"\nThe tool found the exponential decay law from raw measurements.")
