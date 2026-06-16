import numpy as np
import pandas as pd

def run_monte_carlo_simulation(S0, mu, sigma, T, steps, paths):
    """
    Simulates asset price paths using Geometric Brownian Motion (GBM).
    """
    dt = T / steps
    # Generate random shocks
    Z = np.random.standard_normal((steps, paths))
    
    # Path matrix
    path_matrix = np.zeros((steps + 1, paths))
    path_matrix[0] = S0
    
    for t in range(1, steps + 1):
        path_matrix[t] = path_matrix[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])
        
    return path_matrix

def simulate_delta_hedging(path_matrix, K, T, r, sigma):
    """
    Backtests a simple continuous Delta hedging strategy over the generated paths.
    """
    from black_scholes import BlackScholesEngine
    steps, paths = path_matrix.shape
    steps -= 1 # Adjusting for T0 initial step
    dt = T / steps
    
    print(f"Backtesting Delta-hedging constraints over {paths} simulated paths...")
    # Mock calculation sample demonstrating portfolio tracking loop
    final_pnl_errors = []
    for p in range(min(paths, 5)): # Check first few paths for logging
        tracking_error = 0
        for t in range(steps):
            S_t = path_matrix[t, p]
            time_left = T - (t * dt)
            
            # Extract Delta dynamically from the engineering matrix
            engine = BlackScholesEngine(S_t, K, time_left, r, sigma, "call")
            greeks = engine.calculate_greeks()
            current_delta = greeks["Delta"]
            
            # Simple tracking error variation approximation
            tracking_error += current_delta * (path_matrix[t+1, p] - S_t)
            
        final_pnl_errors.append(tracking_error)
        
    print(f"Simulations finished. Mean Tracking error proxy for sample: {np.mean(final_pnl_errors):.4f}")
    return final_pnl_errors

if __name__ == "__main__":
    # Mock parameters
    sim_paths = run_monte_carlo_simulation(S0=100, mu=0.08, sigma=0.2, T=0.5, steps=100, paths=50)
    simulate_delta_hedging(sim_paths, K=100, T=0.5, r=0.05, sigma=0.2)
