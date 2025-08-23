"""
Vasicek Interest Rate Model Implementation

This module implements the Vasicek model with analytical solutions, Monte Carlo simulation,
and parameter estimation methods.

The Vasicek model follows the SDE:
dr(t) = κ(θ - r(t))dt + σdW(t)

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
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings


class VasicekModel:
    """
    Vasicek Interest Rate Model
    
    This class implements the Vasicek model with analytical solutions for bond pricing,
    Monte Carlo simulation, and parameter estimation.
    """
    
    def __init__(self, kappa: float, theta: float, sigma: float, r0: float = None):
        """
        Initialize Vasicek model parameters
        
        Parameters:
        -----------
        kappa : float
            Speed of mean reversion (must be positive)
        theta : float
            Long-term mean level
        sigma : float
            Volatility parameter (must be positive)
        r0 : float, optional
            Initial interest rate (if None, will be set to theta)
        """
        if kappa <= 0 or sigma <= 0:
            raise ValueError("Kappa and sigma must be positive")
        
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0 if r0 is not None else theta
        
    def simulate_path(self, T: float, dt: float = 1/252, n_paths: int = 1, 
                     method: str = 'exact') -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Vasicek process paths using Monte Carlo
        
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
                         self.sigma * dW
        
        return times, rates
    
    def _simulate_exact(self, times: np.ndarray, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """Exact simulation using Ornstein-Uhlenbeck process (optimized)"""
        dt = times[1] - times[0]
        n_steps = len(times)
        
        rates = np.zeros((n_paths, n_steps))
        rates[:, 0] = self.r0
        
        # Pre-compute constants for efficiency
        exp_kappa_dt = np.exp(-self.kappa * dt)
        variance = self.sigma**2 * (1 - exp_kappa_dt**2) / (2 * self.kappa)
        std_dev = np.sqrt(variance)
        
        # Use vectorized operations where possible
        for i in range(1, n_steps):
            # Exact solution for Ornstein-Uhlenbeck process
            mean = self.theta + (rates[:, i-1] - self.theta) * exp_kappa_dt
            
            # Generate from normal distribution (vectorized)
            rates[:, i] = np.random.normal(mean, std_dev, n_paths)
        
        return times, rates
    
    def bond_price(self, r: float, T: float, t: float = 0) -> float:
        """
        Calculate zero-coupon bond price using Vasicek analytical solution
        
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
        
        # Handle edge cases
        if tau <= 0:
            return 1.0
        
        # A and B functions for Vasicek model
        if self.kappa > 1e-10:  # Avoid division by zero
            B = (1 - np.exp(-self.kappa * tau)) / self.kappa
        else:
            B = tau  # Limit as kappa approaches zero
        
        # Improved A function calculation with better numerical stability
        if self.kappa > 1e-10:
            A = np.exp((self.theta - self.sigma**2 / (2 * self.kappa**2)) * 
                       (B - tau) - self.sigma**2 * B**2 / (4 * self.kappa))
        else:
            # Limit as kappa approaches zero
            A = np.exp(-self.theta * tau + self.sigma**2 * tau**3 / 6)
        
        # Ensure bond price is positive and reasonable
        bond_price = A * np.exp(-B * r)
        return max(bond_price, 1e-10)  # Prevent negative or zero prices
    
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
        # Handle edge cases
        valid_maturities = np.where(maturities > 0, maturities, 1e-10)
        
        # Vectorized calculation of A and B functions
        if self.kappa > 1e-10:
            B = (1 - np.exp(-self.kappa * valid_maturities)) / self.kappa
        else:
            B = valid_maturities
        
        # A function (vectorized)
        if self.kappa > 1e-10:
            A = np.exp((self.theta - self.sigma**2 / (2 * self.kappa**2)) * 
                       (B - valid_maturities) - self.sigma**2 * B**2 / (4 * self.kappa))
        else:
            A = np.exp(-self.theta * valid_maturities + self.sigma**2 * valid_maturities**3 / 6)
        
        # Bond prices
        prices = A * np.exp(-B * r)
        return np.maximum(prices, 1e-10)  # Ensure positive prices
    
    def forward_rate(self, r: float, T: float, t: float = 0) -> float:
        """
        Calculate instantaneous forward rate
        
        Parameters:
        -----------
        r : float
            Current interest rate
        T : float
            Forward rate maturity
        t : float
            Current time (default: 0)
            
        Returns:
        --------
        float
            Forward rate
        """
        tau = T - t
        B = (1 - np.exp(-self.kappa * tau)) / self.kappa
        
        # Forward rate formula
        forward = r * np.exp(-self.kappa * tau) + \
                 self.theta * (1 - np.exp(-self.kappa * tau)) - \
                 self.sigma**2 * B * np.exp(-self.kappa * tau) / (2 * self.kappa)
        
        return forward
    
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
        if dt <= 0:
            return 1e-10
        
        try:
            exp_kappa_dt = np.exp(-self.kappa * dt)
            mean = self.theta + (r_s - self.theta) * exp_kappa_dt
            variance = self.sigma**2 * (1 - exp_kappa_dt**2) / (2 * self.kappa)
            
            # Handle numerical issues
            if variance <= 0:
                return 1e-10
            
            # Normal density
            density = stats.norm.pdf(r_t, mean, np.sqrt(variance))
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
        Estimate Vasicek parameters from observed data
        
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
            if kappa <= 0 or sigma <= 0:
                return np.inf
            
            try:
                # Create temporary model for likelihood calculation
                temp_model = VasicekModel(kappa, theta, sigma)
                ll = temp_model.log_likelihood(rates, dt)
                return -ll if np.isfinite(ll) else np.inf
            except (ValueError, RuntimeWarning, OverflowError):
                return np.inf
        
        # Initial guess using moment estimates
        try:
            kappa_init, theta_init, sigma_init = self._estimate_moment(rates, dt)
            # Ensure initial values are reasonable
            kappa_init = max(0.01, min(10, kappa_init))
            theta_init = max(-0.5, min(0.5, theta_init))
            sigma_init = max(0.01, min(1, sigma_init))
        except:
            # Fallback to reasonable defaults
            kappa_init, theta_init, sigma_init = 0.1, 0.05, 0.1
        
        try:
            result = minimize(neg_log_likelihood, [kappa_init, theta_init, sigma_init],
                            method='L-BFGS-B', bounds=[(0.01, 10), (-0.5, 0.5), (0.01, 1)])
            
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
            theta = max(-0.1, min(0.2, theta))   # More reasonable theta bounds for Vasicek
            sigma = max(0.001, min(0.5, sigma))  # More reasonable sigma bounds
            
            return kappa, theta, sigma
        except Exception as e:
            # Return reasonable defaults if estimation fails
            return 0.1, 0.05, 0.1
    
    def stationary_distribution(self) -> Tuple[float, float]:
        """
        Calculate stationary distribution parameters
        
        Returns:
        --------
        tuple
            (mean, variance) of stationary distribution
        """
        mean = self.theta
        variance = self.sigma**2 / (2 * self.kappa)
        return mean, variance
    
    def plot_simulation(self, times: np.ndarray, rates: np.ndarray, 
                       title: str = "Vasicek Model Simulation"):
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
        
        # Plot theoretical stationary distribution
        stat_mean, stat_var = self.stationary_distribution()
        x = np.linspace(min(final_rates), max(final_rates), 100)
        plt.plot(x, stats.norm.pdf(x, stat_mean, np.sqrt(stat_var)), 
                'r-', label='Stationary distribution')
        plt.axvline(self.theta, color='g', linestyle='--', label='Long-term mean')
        plt.xlabel('Interest Rate')
        plt.ylabel('Density')
        plt.title('Final Rate Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()
    
    def create_sample_data(self, n_obs: int = 1000, dt: float = 1/252, **kwargs) -> np.ndarray:
        """Create sample interest rate data using Vasicek model"""
        kappa = kwargs.get('kappa', self.kappa)
        theta = kwargs.get('theta', self.theta)
        sigma = kwargs.get('sigma', self.sigma)
        r0 = kwargs.get('r0', self.r0)
        
        model = VasicekModel(kappa, theta, sigma, r0)
        times, rates = model.simulate_path(n_obs * dt, dt, n_paths=1)
        return rates[0, :]
    
    def compare_with_cir(self, cir_model, times: np.ndarray, n_paths: int = 1000):
        """
        Compare Vasicek and CIR model simulations
        
        Parameters:
        -----------
        cir_model : CIRModel
            CIR model instance
        times : np.ndarray
            Time points
        n_paths : int
            Number of simulation paths
        """
        # Simulate both models
        _, vasicek_rates = self.simulate_path(times[-1], times[1]-times[0], n_paths, 'exact')
        _, cir_rates = cir_model.simulate_path(times[-1], times[1]-times[0], n_paths, 'exact')
        
        plt.figure(figsize=(15, 10))
        
        # Compare paths
        plt.subplot(2, 3, 1)
        for i in range(min(5, n_paths)):
            plt.plot(times, vasicek_rates[i, :], 'b-', alpha=0.7, label='Vasicek' if i == 0 else "")
            plt.plot(times, cir_rates[i, :], 'r-', alpha=0.7, label='CIR' if i == 0 else "")
        plt.xlabel('Time (years)')
        plt.ylabel('Interest Rate')
        plt.title('Sample Paths Comparison')
        plt.legend()
        plt.grid(True)
        
        # Compare means
        plt.subplot(2, 3, 2)
        vasicek_mean = np.mean(vasicek_rates, axis=0)
        cir_mean = np.mean(cir_rates, axis=0)
        plt.plot(times, vasicek_mean, 'b-', label='Vasicek')
        plt.plot(times, cir_mean, 'r-', label='CIR')
        plt.xlabel('Time (years)')
        plt.ylabel('Mean Rate')
        plt.title('Mean Comparison')
        plt.legend()
        plt.grid(True)
        
        # Compare volatilities
        plt.subplot(2, 3, 3)
        vasicek_std = np.std(vasicek_rates, axis=0)
        cir_std = np.std(cir_rates, axis=0)
        plt.plot(times, vasicek_std, 'b-', label='Vasicek')
        plt.plot(times, cir_std, 'r-', label='CIR')
        plt.xlabel('Time (years)')
        plt.ylabel('Volatility')
        plt.title('Volatility Comparison')
        plt.legend()
        plt.grid(True)
        
        # Compare yield curves
        plt.subplot(2, 3, 4)
        maturities = np.linspace(0.1, 10, 50)
        current_rate = vasicek_mean[-1]
        vasicek_yields = self.yield_curve(current_rate, maturities)
        cir_yields = cir_model.yield_curve(current_rate, maturities)
        plt.plot(maturities, vasicek_yields, 'b-', label='Vasicek')
        plt.plot(maturities, cir_yields, 'r-', label='CIR')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield')
        plt.title('Yield Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        # Compare final distributions
        plt.subplot(2, 3, 5)
        plt.hist(vasicek_rates[:, -1], bins=30, density=True, alpha=0.7, 
                color='blue', label='Vasicek')
        plt.hist(cir_rates[:, -1], bins=30, density=True, alpha=0.7, 
                color='red', label='CIR')
        plt.xlabel('Interest Rate')
        plt.ylabel('Density')
        plt.title('Final Distribution Comparison')
        plt.legend()
        plt.grid(True)
        
        # Compare bond prices
        plt.subplot(2, 3, 6)
        maturities = np.array([0.5, 1, 2, 5, 10])
        current_rate = vasicek_mean[-1]
        vasicek_prices = [self.bond_price(current_rate, T) for T in maturities]
        cir_prices = [cir_model.bond_price(current_rate, T) for T in maturities]
        plt.plot(maturities, vasicek_prices, 'bo-', label='Vasicek')
        plt.plot(maturities, cir_prices, 'ro-', label='CIR')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Bond Price')
        plt.title('Bond Price Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle('Vasicek vs CIR Model Comparison', y=1.02)
        plt.show()


def create_sample_data(n_obs: int = 1000, dt: float = 1/252, 
                      kappa: float = 0.1, theta: float = 0.05, 
                      sigma: float = 0.02, r0: float = 0.05) -> np.ndarray:
    """
    Create sample interest rate data using Vasicek model
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    dt : float
        Time step
    kappa, theta, sigma, r0 : float
        Vasicek model parameters
        
    Returns:
    --------
    np.ndarray
        Simulated interest rates
    """
    model = VasicekModel(kappa, theta, sigma, r0)
    times, rates = model.simulate_path(n_obs * dt, dt, n_paths=1)
    return rates[0, :]


if __name__ == "__main__":
    # Example usage
    print("Vasicek Model Example")
    print("=" * 50)
    
    # Create model
    kappa, theta, sigma = 0.1, 0.05, 0.02
    model = VasicekModel(kappa, theta, sigma, r0=0.03)
    
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
    
    # Stationary distribution
    stat_mean, stat_var = model.stationary_distribution()
    print(f"Stationary distribution: mean={stat_mean:.4f}, variance={stat_var:.6f}")
