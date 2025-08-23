"""
Cox-Ingersoll-Ross (CIR) Interest Rate Model Implementation

This module implements the CIR model with analytical solutions, Monte Carlo simulation,
and parameter estimation methods.

The CIR model follows the SDE:
dr(t) = κ(θ - r(t))dt + σ√r(t)dW(t)

Where:
- r(t): instantaneous interest rate
- κ: speed of mean reversion (kappa)
- θ: long-term mean level (theta)
- σ: volatility parameter (sigma)
- W(t): Wiener process
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma, kv
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings


class CIRModel:
    """
    Cox-Ingersoll-Ross (CIR) Interest Rate Model
    
    This class implements the CIR model with analytical solutions for bond pricing,
    Monte Carlo simulation, and parameter estimation.
    """
    
    def __init__(self, kappa: float, theta: float, sigma: float, r0: float = None):
        """
        Initialize CIR model parameters
        
        Parameters:
        -----------
        kappa : float
            Speed of mean reversion (must be positive)
        theta : float
            Long-term mean level (must be positive)
        sigma : float
            Volatility parameter (must be positive)
        r0 : float, optional
            Initial interest rate (if None, will be set to theta)
        """
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            raise ValueError("All parameters must be positive")
        
        # Feller condition: 2*κ*θ ≥ σ²
        if 2 * kappa * theta < sigma**2:
            warnings.warn("Feller condition violated: 2*κ*θ < σ². Process may hit zero.")
        
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0 if r0 is not None else theta
        
    def simulate_path(self, T: float, dt: float = 1/252, n_paths: int = 1, 
                     method: str = 'euler') -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate CIR process paths using Monte Carlo
        
        Parameters:
        -----------
        T : float
            Time horizon in years
        dt : float
            Time step size (default: daily)
        n_paths : int
            Number of simulation paths
        method : str
            Simulation method ('euler' or 'exact')
            
        Returns:
        --------
        times : np.ndarray
            Time points
        rates : np.ndarray
            Simulated interest rates (n_paths x n_steps)
        """
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        
        if method == 'euler':
            return self._simulate_euler(times, n_paths)
        elif method == 'exact':
            return self._simulate_exact(times, n_paths)
        else:
            raise ValueError("Method must be 'euler' or 'exact'")
    
    def _simulate_euler(self, times: np.ndarray, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """Euler-Maruyama discretization"""
        dt = times[1] - times[0]
        n_steps = len(times)
        
        rates = np.zeros((n_paths, n_steps))
        rates[:, 0] = self.r0
        
        for i in range(1, n_steps):
            # Generate random increments
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            
            # Euler-Maruyama scheme
            rates[:, i] = rates[:, i-1] + self.kappa * (self.theta - rates[:, i-1]) * dt + \
                         self.sigma * np.sqrt(np.maximum(rates[:, i-1], 0)) * dW
            
            # Ensure non-negativity
            rates[:, i] = np.maximum(rates[:, i], 0)
        
        return times, rates
    
    def _simulate_exact(self, times: np.ndarray, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """Exact simulation using non-central chi-squared distribution (optimized)"""
        dt = times[1] - times[0]
        n_steps = len(times)
        
        rates = np.zeros((n_paths, n_steps))
        rates[:, 0] = self.r0
        
        # Pre-compute constants for efficiency
        c = self.sigma**2 * (1 - np.exp(-self.kappa * dt)) / (4 * self.kappa)
        df = 4 * self.kappa * self.theta / self.sigma**2
        exp_kappa_dt = np.exp(-self.kappa * dt)
        
        # Use vectorized operations where possible
        for i in range(1, n_steps):
            # Parameters for non-central chi-squared distribution
            nc = rates[:, i-1] * exp_kappa_dt / c
            
            # Generate from non-central chi-squared (vectorized)
            rates[:, i] = c * np.random.noncentral_chisquare(df, nc, n_paths)
        
        return times, rates
    
    def bond_price(self, r: float, T: float, t: float = 0) -> float:
        """
        Calculate zero-coupon bond price using CIR analytical solution
        
        Parameters:
        -----------
        r : float
            Current interest rate
        T : float
            Bond maturity
        t : float
            Current time (default: 0)
            
        Returns:
        --------
        float
            Bond price
        """
        tau = T - t
        gamma = np.sqrt(self.kappa**2 + 2 * self.sigma**2)
        
        # A and B functions
        B = 2 * (np.exp(gamma * tau) - 1) / ((gamma + self.kappa) * (np.exp(gamma * tau) - 1) + 2 * gamma)
        A = (2 * gamma * np.exp((self.kappa + gamma) * tau / 2) / 
             ((gamma + self.kappa) * (np.exp(gamma * tau) - 1) + 2 * gamma)) ** (2 * self.kappa * self.theta / self.sigma**2)
        
        return A * np.exp(-B * r)
    
    def yield_curve(self, r: float, maturities: np.ndarray) -> np.ndarray:
        """
        Calculate yield curve for given maturities (vectorized)
        
        Parameters:
        -----------
        r : float
            Current interest rate
        maturities : np.ndarray
            Array of maturities in years
            
        Returns:
        --------
        np.ndarray
            Yields for each maturity
        """
        prices = self.bond_prices_vectorized(r, maturities)
        yields = -np.log(prices) / maturities
        return yields
    
    def bond_prices_vectorized(self, r: float, maturities: np.ndarray) -> np.ndarray:
        """
        Calculate bond prices for multiple maturities (vectorized for performance)
        
        Parameters:
        -----------
        r : float
            Current interest rate
        maturities : np.ndarray
            Array of maturities in years
            
        Returns:
        --------
        np.ndarray
            Bond prices for each maturity
        """
        # Vectorized calculation of A and B functions
        gamma = np.sqrt(self.kappa**2 + 2 * self.sigma**2)
        
        # B function (vectorized)
        exp_gamma_tau = np.exp(gamma * maturities)
        B = 2 * (exp_gamma_tau - 1) / ((gamma + self.kappa) * (exp_gamma_tau - 1) + 2 * gamma)
        
        # A function (vectorized)
        exp_term = np.exp((self.kappa + gamma) * maturities / 2)
        A = (2 * gamma * exp_term / 
             ((gamma + self.kappa) * (exp_gamma_tau - 1) + 2 * gamma)) ** (2 * self.kappa * self.theta / self.sigma**2)
        
        # Bond prices
        prices = A * np.exp(-B * r)
        return prices
    
    def transition_density(self, r_t: float, r_s: float, dt: float) -> float:
        """
        Calculate transition density P(r_t | r_s) for time step dt
        
        Parameters:
        -----------
        r_t : float
            Interest rate at time t
        r_s : float
            Interest rate at time s
        dt : float
            Time step (t - s)
            
        Returns:
        --------
        float
            Transition density
        """
        # Handle edge cases
        if r_t <= 0 or r_s <= 0 or dt <= 0:
            return 1e-10  # Small positive value instead of 0
        
        c = self.sigma**2 * (1 - np.exp(-self.kappa * dt)) / (4 * self.kappa)
        q = 2 * self.kappa * self.theta / self.sigma**2 - 1
        u = r_s * np.exp(-self.kappa * dt) / c
        v = r_t / c
        
        # Handle numerical issues
        if c <= 0 or u <= 0 or v <= 0:
            return 1e-10
        
        try:
            # Non-central chi-squared density
            density = (1 / c) * np.exp(-u - v) * (v / u)**(q/2) * kv(q, 2 * np.sqrt(u * v))
            return max(density, 1e-10)  # Ensure positive value
        except (ValueError, OverflowError, RuntimeWarning):
            return 1e-10
    
    def log_likelihood(self, rates: np.ndarray, dt: float) -> float:
        """
        Calculate log-likelihood for parameter estimation
        
        Parameters:
        -----------
        rates : np.ndarray
            Observed interest rates
        dt : float
            Time step between observations
            
        Returns:
        --------
        float
            Log-likelihood
        """
        ll = 0.0
        for i in range(1, len(rates)):
            density = self.transition_density(rates[i], rates[i-1], dt)
            if density > 0:
                ll += np.log(density)
            else:
                ll += np.log(1e-10)  # Use small positive value for log
        return ll
    
    def estimate_parameters(self, rates: np.ndarray, dt: float, 
                          method: str = 'mle') -> Tuple[float, float, float]:
        """
        Estimate CIR parameters from observed data
        
        Parameters:
        -----------
        rates : np.ndarray
            Observed interest rates
        dt : float
            Time step between observations
        method : str
            Estimation method ('mle' or 'moment')
            
        Returns:
        --------
        tuple
            Estimated (kappa, theta, sigma)
        """
        if method == 'mle':
            return self._estimate_mle(rates, dt)
        elif method == 'moment':
            return self._estimate_moment(rates, dt)
        else:
            raise ValueError("Method must be 'mle' or 'moment'")
    
    def _estimate_mle(self, rates: np.ndarray, dt: float) -> Tuple[float, float, float]:
        """Maximum likelihood estimation"""
        def neg_log_likelihood(params):
            kappa, theta, sigma = params
            if kappa <= 0 or theta <= 0 or sigma <= 0:
                return np.inf
            
            try:
                # Create temporary model for likelihood calculation
                temp_model = CIRModel(kappa, theta, sigma)
                ll = temp_model.log_likelihood(rates, dt)
                return -ll if np.isfinite(ll) else np.inf
            except (ValueError, RuntimeWarning, OverflowError):
                return np.inf
        
        # Initial guess using moment estimates
        try:
            kappa_init, theta_init, sigma_init = self._estimate_moment(rates, dt)
            # Ensure initial values are reasonable
            kappa_init = max(0.01, min(10, kappa_init))
            theta_init = max(0.01, min(0.5, theta_init))
            sigma_init = max(0.01, min(1, sigma_init))
        except:
            # Fallback to reasonable defaults
            kappa_init, theta_init, sigma_init = 0.1, 0.05, 0.1
        
        try:
            result = minimize(neg_log_likelihood, [kappa_init, theta_init, sigma_init],
                            method='L-BFGS-B', bounds=[(0.01, 10), (0.01, 0.5), (0.01, 1)])
            
            if result.success:
                return result.x
            else:
                # If optimization fails, return moment estimates
                return self._estimate_moment(rates, dt)
        except:
            # If all else fails, return moment estimates
            return self._estimate_moment(rates, dt)
    
    def _estimate_moment(self, rates: np.ndarray, dt: float) -> Tuple[float, float, float]:
        """Improved moment matching estimation"""
        try:
            # Calculate sample moments
            mean_rate = np.mean(rates)
            var_rate = np.var(rates)
            
            # Calculate autocorrelation with better handling
            if len(rates) > 1:
                autocorr = np.corrcoef(rates[:-1], rates[1:])[0, 1]
                if np.isnan(autocorr) or autocorr <= 0 or autocorr >= 1:
                    # Use alternative estimation method
                    rate_changes = np.diff(rates)
                    autocorr = np.corrcoef(rate_changes[:-1], rate_changes[1:])[0, 1]
                    if np.isnan(autocorr) or autocorr <= 0 or autocorr >= 1:
                        autocorr = 0.5  # Default value
            else:
                autocorr = 0.5
            
            # Improved parameter estimation
            kappa = -np.log(max(autocorr, 0.01)) / dt  # Ensure positive kappa
            theta = mean_rate
            sigma = np.sqrt(2 * kappa * var_rate)
            
            # Ensure reasonable bounds with better constraints
            kappa = max(0.001, min(5.0, kappa))  # More reasonable kappa bounds
            theta = max(0.001, min(0.2, theta))  # More reasonable theta bounds
            sigma = max(0.001, min(0.5, sigma))  # More reasonable sigma bounds
            
            # Check Feller condition
            if 2 * kappa * theta < sigma**2:
                # Adjust sigma to satisfy Feller condition
                sigma = np.sqrt(1.8 * kappa * theta)  # Leave some margin
            
            return kappa, theta, sigma
        except Exception as e:
            # Return reasonable defaults if estimation fails
            return 0.1, 0.05, 0.1
    
    def plot_simulation(self, times: np.ndarray, rates: np.ndarray, 
                       title: str = "CIR Model Simulation"):
        """Plot simulation results"""
        plt.figure(figsize=(12, 8))
        
        # Plot interest rate paths
        plt.subplot(2, 2, 1)
        for i in range(min(10, rates.shape[0])):  # Plot first 10 paths
            plt.plot(times, rates[i, :], alpha=0.7)
        plt.plot(times, self.theta * np.ones_like(times), 'r--', label='Long-term mean')
        plt.xlabel('Time (years)')
        plt.ylabel('Interest Rate')
        plt.title('Interest Rate Paths')
        plt.legend()
        plt.grid(True)
        
        # Plot mean and confidence intervals
        plt.subplot(2, 2, 2)
        mean_rate = np.mean(rates, axis=0)
        std_rate = np.std(rates, axis=0)
        plt.plot(times, mean_rate, 'b-', label='Mean')
        plt.fill_between(times, mean_rate - 2*std_rate, mean_rate + 2*std_rate, 
                        alpha=0.3, label='95% CI')
        plt.plot(times, self.theta * np.ones_like(times), 'r--', label='Long-term mean')
        plt.xlabel('Time (years)')
        plt.ylabel('Interest Rate')
        plt.title('Mean and Confidence Intervals')
        plt.legend()
        plt.grid(True)
        
        # Plot yield curve
        plt.subplot(2, 2, 3)
        maturities = np.linspace(0.1, 10, 50)
        current_rate = mean_rate[-1]  # Use final mean rate
        yields = self.yield_curve(current_rate, maturities)
        plt.plot(maturities, yields)
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield')
        plt.title('Yield Curve')
        plt.grid(True)
        
        # Plot distribution
        plt.subplot(2, 2, 4)
        final_rates = rates[:, -1]
        plt.hist(final_rates, bins=30, density=True, alpha=0.7)
        plt.axvline(self.theta, color='r', linestyle='--', label='Long-term mean')
        plt.xlabel('Interest Rate')
        plt.ylabel('Density')
        plt.title('Final Rate Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()
    
    def create_sample_data(self, n_obs: int = 1000, dt: float = 1/252, **kwargs) -> np.ndarray:
        """Create sample interest rate data using CIR model"""
        kappa = kwargs.get('kappa', self.kappa)
        theta = kwargs.get('theta', self.theta)
        sigma = kwargs.get('sigma', self.sigma)
        r0 = kwargs.get('r0', self.r0)
        
        model = CIRModel(kappa, theta, sigma, r0)
        times, rates = model.simulate_path(n_obs * dt, dt, n_paths=1)
        return rates[0, :]


def create_sample_data(n_obs: int = 1000, dt: float = 1/252, 
                      kappa: float = 0.1, theta: float = 0.05, 
                      sigma: float = 0.1, r0: float = 0.05) -> np.ndarray:
    """
    Create sample interest rate data using CIR model
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    dt : float
        Time step
    kappa, theta, sigma, r0 : float
        CIR model parameters
        
    Returns:
    --------
    np.ndarray
        Simulated interest rates
    """
    model = CIRModel(kappa, theta, sigma, r0)
    times, rates = model.simulate_path(n_obs * dt, dt, n_paths=1)
    return rates[0, :]


if __name__ == "__main__":
    # Example usage
    print("CIR Model Example")
    print("=" * 50)
    
    # Create model
    kappa, theta, sigma = 0.1, 0.05, 0.1
    model = CIRModel(kappa, theta, sigma, r0=0.03)
    
    # Simulate paths
    times, rates = model.simulate_path(T=5, dt=1/252, n_paths=1000, method='exact')
    
    # Plot results
    model.plot_simulation(times, rates)
    
    # Calculate bond price
    bond_price = model.bond_price(0.05, 5)
    print(f"5-year bond price with r=5%: {bond_price:.4f}")
    
    # Calculate yield curve
    maturities = np.array([0.5, 1, 2, 5, 10])
    yields = model.yield_curve(0.05, maturities)
    print(f"Yield curve: {yields}")
