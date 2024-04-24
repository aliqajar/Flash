import torch
import time
import numpy as np

def two_phase_flash_calculation(z, K, P, T):
    """
    Perform two-phase flash calculation for a thermodynamic system using PyTorch.
    
    Parameters:
    - z: mole fractions of components (PyTorch tensor)
    - K: equilibrium ratios (PyTorch tensor, num_components)
    - P: pressure (scalar)
    - T: temperature (scalar)
    
    Returns:
    - x: mole fractions of liquid phase (PyTorch tensor, num_components)
    - y: mole fractions of vapor phase (PyTorch tensor, num_components)
    - V: vapor fraction (PyTorch tensor, scalar)
    """
    
    # Initialize variables
    max_iter = 10
    tolerance = 1e-3
    num_components = z.shape[0]
    
    # Initialize vapor fraction
    V = torch.tensor(0.5, device=z.device, requires_grad=True)
    
    # Iterate until convergence or maximum iterations reached
    for i in range(max_iter):
        # Calculate liquid and vapor mole fractions
        x = z / (1 + V * (K - 1))
        y = K * x
        
        # Calculate function f and its derivative df
        f = torch.sum(z * (K - 1) / (1 + V * (K - 1)))
        df = -torch.sum(z * (K - 1)**2 / (1 + V * (K - 1))**2)
        
        # Update vapor fraction using Newton-Raphson method
        V_new = V - f / df
        
        # Check for convergence
        if torch.abs(V_new - V) < tolerance:
            break
        
        V = V_new
    
    return x, y, V

# Set a different seed value based on the current time
seed_value = int(time.time())
torch.manual_seed(seed_value)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate random mole fractions for 1000 components
num_components = 1_000_000
z = torch.rand(num_components, device=device)
z /= torch.sum(z)  # Normalize mole fractions to sum up to 1

beta = torch.normal(mean=0, std=1.0, size=(num_components,), device=device)
K = torch.exp(beta)

P = 1.0  # Pressure (bar)
T = 300.0  # Temperature (K)

x, y, V = two_phase_flash_calculation(z, K, P, T)

# Convert the results to NumPy arrays
x_np = x.detach().cpu().numpy()
y_np = y.detach().cpu().numpy()

# Set NumPy print options to display all values and 10 columns per row
np.set_printoptions(threshold=np.inf, linewidth=200)

print("Seed value:", seed_value)
# print("Liquid mole fractions (x):")
# print(x.detach().cpu().numpy())
# print("Vapor mole fractions (y):")
# print(y.detach().cpu().numpy())
print("Vapor fraction (V):", V.detach().cpu().item())

# Print negative liquid mole fractions
negative_x = x_np[x_np < 0]
if negative_x.size > 0:
    print("Negative liquid mole fractions (x):")
    print(negative_x)
else:
    print("No negative liquid mole fractions found.")

# Print negative vapor mole fractions
negative_y = y_np[y_np < 0]
if negative_y.size > 0:
    print("Negative vapor mole fractions (y):")
    print(negative_y)
else:
    print("No negative vapor mole fractions found.")