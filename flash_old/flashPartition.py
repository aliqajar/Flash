

import numpy as np
from scipy.optimize import fsolve

# Function to calculate the vapor fraction using the Rachford-Rice equation
def calculate_vapor_fraction(z, K):
    def objective_function(V):
        return np.sum(z * (K - 1) / (1 + V * (K - 1)))
    # Initial guess for vapor fraction
    V_guess = 0.5
    # Solve for V
    V, = fsolve(objective_function, V_guess)
    # Ensure V is within physical bounds
    V = max(min(V, 1), 0)
    return V

# Function to perform flash calculations
def flash_calculation(z, K):
    # Calculate vapor fraction
    V = calculate_vapor_fraction(z, K)
    # Calculate phase compositions
    x = z / (1 + V * (K - 1))
    y = K * x
    # Return results
    return V, x, y

# Example usage:
# Overall composition (z)
z = np.array([0.4, 0.6])  # Example for a binary system
# K-values for each component
K = np.array([2.0, 0.5])  # Example K-values

# Perform flash calculation
V, x, y = flash_calculation(z, K)

print(f"Vapor fraction (V): {V}")
print(f"Liquid phase composition (x): {x}")
print(f"Vapor phase composition (y): {y}")
