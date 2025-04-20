import numpy as np
import matplotlib.pyplot as plt
dividend = 0.02
strike_price = 100
volatility = 0.2
risk_free_rate = 0.05
time_to_maturity = 1
n_timesteps=500
n_spacesteps=1000
alpha=0.5
# Parameters
x_domain = np.linspace(0, 500, n_spacesteps)  # Asset prices
t_domain = np.linspace(0, 1, n_timesteps)  # Time from 0 to T
time_interval = 1.0 / n_timesteps

# Initial condition for v at T (payoff for an American call option)
v_T = np.maximum(x_domain - strike_price, 0)
lambda_T = np.zeros(len(x_domain))

# Matrix A setup for the finite difference method
A = np.zeros((len(x_domain), len(x_domain)))
for i in range(len(x_domain)):
    A[i, i] = - ((volatility * i)**2 + risk_free_rate)
    x_value = x_domain[i]
    if i > 0:
        A[i, i - 1] = 0.5 * (((volatility * i)**2 - (risk_free_rate - dividend) * i))
    if i < len(x_domain) - 1:
        A[i, i + 1] = 0.5 * (((volatility * i)**2 + (risk_free_rate - dividend) * i))

# Storage array for the option values at each time step
v_matrix = np.zeros((len(t_domain), len(x_domain)))
v_matrix[-1, :] = v_T  # Set the terminal condition at t = T

# Backward time-stepping
for i in range(len(t_domain) - 2, -1, -1):
    t = t_domain[i]
    
    # Step 1: LU decomposition to find the intermediate step
    A_prime = -1 / time_interval * np.eye(len(x_domain)) + alpha * A
    B = 1 / time_interval * np.eye(len(x_domain)) + (1 - alpha) * A
    intermediate = np.linalg.solve(A_prime, -np.dot(B, v_T) - lambda_T)

    # Step 2: Projection step to enforce the early-exercise constraint
    v_new = np.copy(intermediate)  # Start with the intermediate values as a base
    for j in range(len(x_domain)):
        # Calculate the payoff at this point (for an American call, payoff = max(x - strike_price, 0))
        payoff = max(x_domain[j] - strike_price, 0)
        # Enforce the constraint: v >= payoff
        if v_new[j] + time_interval * lambda_T[j] < payoff:
            v_new[j] = payoff
            lambda_T[j] = max(lambda_T[j] + (1 / time_interval) * (intermediate[j] - v_new[j]), 0)
        else:
            v_new[j] = v_new[j] + time_interval * lambda_T[j]
            lambda_T[j] = 0
    # Update v_T for the next time step
    v_T = v_new
    # Store the new v_T in the matrix for plotting
    v_matrix[i, :] = v_T
# Define the time and asset price ranges
time_steps = v_matrix.shape[0]
asset_prices = v_matrix.shape[1]

# Create linearly spaced arrays for time and asset prices
t = np.linspace(0, 1, time_steps)
S = np.linspace(0, 500, asset_prices)

# Create a meshgrid
T, S_mesh = np.meshgrid(t, S)

# Transpose v_matrix to match the meshgrid dimensions
V = v_matrix.T
# Create a figure and a 3D axis
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(T, S_mesh, V, cmap='viridis', edgecolor='none')

# Add labels and title
ax.set_xlabel("Time (t)")
ax.set_ylabel("Asset Price (S)")
ax.set_zlabel("Option Value (v)")
ax.set_title("American Call Option Value Over Time and Asset Price")

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Display the plot
plt.show()
