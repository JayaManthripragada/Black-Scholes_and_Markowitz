import numpy as np
from scipy.stats import norm

class BlackScholesEngine:
    def __init__(self, S, K, T, r, sigma, option_type="call"):
        """
        S: Spot price, K: Strike price, T: Time to maturity (years),
        r: Risk-free rate, sigma: Volatility, option_type: "call" or "put"
        """
        self.S = float(S)
        self.K = float(K)
        self.T = float(T) if T > 0 else 1e-5 # Prevent division by zero
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        
        # Calculate intermediate d1 and d2 parameters
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def calculate_price(self):
        if self.option_type == "call":
            return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    def calculate_greeks(self):
        """Returns a dictionary of standard Option Greeks."""
        pdf_d1 = norm.pdf(self.d1)
        cdf_d1 = norm.cdf(self.d1)
        cdf_d2 = norm.cdf(self.d2)
        
        # Delta
        delta = cdf_d1 if self.option_type == "call" else cdf_d1 - 1
        
        # Gamma (same for call and put)
        gamma = pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))
        
        # Vega (same for call and put)
        vega = self.S * np.sqrt(self.T) * pdf_d1
        
        # Theta
        theta_base = -(self.S * pdf_d1 * self.sigma) / (2 * np.sqrt(self.T))
        if self.option_type == "call":
            theta = theta_base - self.r * self.K * np.exp(-self.r * self.T) * cdf_d2
        else:
            theta = theta_base + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            
        # Rho
        if self.option_type == "call":
            rho = self.K * self.T * np.exp(-self.r * self.T) * cdf_d2
        else:
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            
        return {"Delta": delta, "Gamma": gamma, "Vega": vega / 100, "Theta": theta / 365, "Rho": rho / 100}

if __name__ == "__main__":
    bs = BlackScholesEngine(S=100, K=105, T=0.5, r=0.05, sigma=0.2, option_type="call")
    print(f"Call Price: {bs.calculate_price():.4f}")
    print("Greeks:", {k: round(v, 5) for k, v in bs.calculate_greeks().items()})
