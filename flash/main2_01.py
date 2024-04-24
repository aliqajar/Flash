import numpy as np

def flash_calculation(z, K, P, T):
    """
    Perform flash calculation for a thermodynamic system.
    
    Parameters:
    - z: mole fractions of components (array)
    - K: equilibrium ratios (array)
    - P: pressure (scalar)
    - T: temperature (scalar)
    
    Returns:
    - x: mole fractions of liquid phase (array)
    - y: mole fractions of vapor phase (array)
    - V: vapor fraction (scalar)
    """
    
    # Initialize variables
    max_iter = 100
    tolerance = 1e-6
    
    # Calculate initial guess for vapor fraction
    V = np.sum(z * (K - 1)) / np.sum(z * (K - 1) + 1)
    
    # Iterate until convergence or maximum iterations reached
    for i in range(max_iter):
        # Calculate liquid and vapor mole fractions
        x = z / (1 + V * (K - 1))
        y = K * x
        
        # Calculate function f and its derivative df
        f = np.sum(z * (K - 1) / (1 + V * (K - 1)))
        df = -np.sum(z * (K - 1)**2 / (1 + V * (K - 1))**2)
        
        # Update vapor fraction using Newton-Raphson method
        V_new = V - f / df
        
        # Check for convergence
        if np.abs(V_new - V) < tolerance:
            break
        
        V = V_new
    
    return x, y, V

# Example usage
z = np.array([0.5, 0.3, 0.2])  # Mole fractions of components
K = np.array([3.0, 1.5, 0.8])  # Equilibrium ratios
P = 1.0  # Pressure (bar)
T = 300.0  # Temperature (K)

x, y, V = flash_calculation(z, K, P, T)

print("Liquid mole fractions (x):", x)
print("Vapor mole fractions (y):", y)
print("Vapor fraction (V):", V)