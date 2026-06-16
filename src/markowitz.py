import numpy as np
import scipy.optimize as sco

class MarkowitzOptimizer:
    def __init__(self, log_returns, risk_free_rate=0.04):
        self.returns = log_returns
        self.rf = risk_free_rate
        self.num_assets = len(log_returns.columns)
        self.mean_returns = log_returns.mean() * 252 # Annualized
        self.cov_matrix = log_returns.cov() * 252   # Annualized

    def _portfolio_stats(self, weights):
        weights = np.array(weights)
        p_ret = np.sum(self.mean_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (p_ret - self.rf) / p_vol
        return p_ret, p_vol, sharpe

    def maximize_sharpe_ratio(self):
        # Objective function to MINIMIZE (negative Sharpe)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(self.num_assets))          # No short selling
        initial_guess = self.num_assets * [1.0 / self.num_assets]
        
        def objective(weights):
            return -self._portfolio_stats(weights)[2]

        result = sco.minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_weights = result.x
        ret, vol, sharpe = self._portfolio_stats(opt_weights)
        
        return {
            "weights": dict(zip(self.returns.columns, opt_weights)),
            "expected_return": ret,
            "volatility": vol,
            "sharpe_ratio": sharpe
        }
