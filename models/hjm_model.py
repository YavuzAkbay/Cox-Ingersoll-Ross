"""
Heath-Jarrow-Morton (HJM) Framework Implementation

This module implements the HJM framework for forward rate modeling, which is one of the most
sophisticated approaches to interest rate modeling. The HJM framework models the entire
forward rate curve as a stochastic process, providing a unified approach to pricing
interest rate derivatives.

The HJM framework is based on the forward rate evolution:
df(t,T) = α(t,T)dt + σ(t,T)dW(t)

Where:
- f(t,T): forward rate at time t for maturity T
- α(t,T): drift function (determined by no-arbitrage conditions)
- σ(t,T): volatility function
- W(t): multi-dimensional Wiener process

Key Features:
- Multi-factor forward rate modeling
- No-arbitrage drift conditions
- Various volatility structures
- Bond pricing and yield curve calculation
- Monte Carlo simulation
- Parameter calibration
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable
import warnings
from datetime import datetime
import seaborn as sns


class HJMForwardRateModel:
    """
    Heath-Jarrow-Morton Forward Rate Model
    
    This class implements the HJM framework for modeling the evolution of forward rates.
    It provides a comprehensive framework for interest rate modeling with multiple factors
    and various volatility structures.
    """
    
    def __init__(self, 
                 volatility_functions: List[Callable],
                 correlation_matrix: Optional[np.ndarray] = None,
                 initial_forward_curve: Optional[np.ndarray] = None,
                 maturities: Optional[np.ndarray] = None):
        """
        Initialize HJM model
        
        Parameters:
        -----------
        volatility_functions : List[Callable]
            List of volatility functions σ_i(t,T) for each factor
        correlation_matrix : np.ndarray, optional
            Correlation matrix between factors (default: identity)
        initial_forward_curve : np.ndarray, optional
            Initial forward rate curve (default: flat curve at 5%)
        maturities : np.ndarray, optional
            Maturity grid for forward rates (default: 0.25 to 30 years)
        """
        self.volatility_functions = volatility_functions
        self.n_factors = len(volatility_functions)
        
        # Set up correlation matrix
        if correlation_matrix is None:
            self.correlation_matrix = np.eye(self.n_factors)
        else:
            if correlation_matrix.shape != (self.n_factors, self.n_factors):
                raise ValueError("Correlation matrix must be square with size n_factors x n_factors")
            self.correlation_matrix = correlation_matrix
        
        # Set up maturity grid
        if maturities is None:
            self.maturities = np.linspace(0.25, 30, 120)  # Quarterly grid
        else:
            self.maturities = maturities
        
        # Set up initial forward curve
        if initial_forward_curve is None:
            self.initial_forward_curve = 0.05 * np.ones_like(self.maturities)
        else:
            if len(initial_forward_curve) != len(self.maturities):
                raise ValueError("Initial forward curve length must match maturities length")
            self.initial_forward_curve = initial_forward_curve
        
        # Pre-compute Cholesky decomposition for correlated Brownian motions
        self.cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)
        
        # Cache for drift calculations
        self._drift_cache = {}
        
    def volatility_function(self, t: float, T: float, factor: int) -> float:
        """
        Evaluate volatility function for given factor
        
        Parameters:
        -----------
        t : float
            Current time
        T : float
            Maturity time
        factor : int
            Factor index
            
        Returns:
        --------
        float
            Volatility value
        """
        if factor >= self.n_factors:
            raise ValueError(f"Factor index {factor} exceeds number of factors {self.n_factors}")
        
        return self.volatility_functions[factor](t, T)
    
    def drift_function(self, t: float, T: float) -> float:
        """
        Calculate HJM drift function (no-arbitrage condition)
        
        The drift is determined by the no-arbitrage condition:
        α(t,T) = Σᵢ σᵢ(t,T) ∫ₜᵀ σᵢ(t,s)ds
        
        Parameters:
        -----------
        t : float
            Current time
        T : float
            Maturity time
            
        Returns:
        --------
        float
            Drift value
        """
        # Check cache first
        cache_key = (t, T)
        if cache_key in self._drift_cache:
            return self._drift_cache[cache_key]
        
        drift = 0.0
        for i in range(self.n_factors):
            # Calculate integral ∫ₜᵀ σᵢ(t,s)ds
            def integrand(s):
                return self.volatility_function(t, s, i)
            
            try:
                integral, _ = quad(integrand, t, T, limit=100)
                drift += self.volatility_function(t, T, i) * integral
            except (ValueError, RuntimeWarning):
                # Handle numerical issues
                drift += 0.0
        
        # Cache the result
        self._drift_cache[cache_key] = drift
        return drift
    
    def simulate_forward_rates(self, 
                              T: float, 
                              dt: float = 1/252, 
                              n_paths: int = 1,
                              method: str = 'euler') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate forward rate evolution using Monte Carlo
        
        Parameters:
        -----------
        T : float
            Time horizon in years
        dt : float
            Time step size
        n_paths : int
            Number of simulation paths
        method : str
            Simulation method ('euler' or 'milstein')
            
        Returns:
        --------
        times : np.ndarray
            Time points
        forward_rates : np.ndarray
            Forward rate surfaces (n_paths x n_times x n_maturities)
        spot_rates : np.ndarray
            Spot rate paths (n_paths x n_times)
        """
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        n_maturities = len(self.maturities)
        
        # Initialize arrays
        forward_rates = np.zeros((n_paths, n_steps + 1, n_maturities))
        spot_rates = np.zeros((n_paths, n_steps + 1))
        
        # Set initial conditions
        forward_rates[:, 0, :] = self.initial_forward_curve
        spot_rates[:, 0] = self.initial_forward_curve[0]  # Spot rate is forward rate at t=0
        
        for path in range(n_paths):
            for i in range(1, n_steps + 1):
                t = times[i-1]
                dt_step = times[i] - times[i-1]
                
                # Generate correlated Brownian increments
                dW = np.random.normal(0, np.sqrt(dt_step), self.n_factors)
                dW_correlated = self.cholesky_matrix @ dW
                
                # Update forward rates for each maturity
                for j, T_maturity in enumerate(self.maturities):
                    if T_maturity > t:  # Only update rates for future maturities
                        # Calculate drift
                        drift = self.drift_function(t, T_maturity)
                        
                        # Calculate diffusion term
                        diffusion = 0.0
                        for k in range(self.n_factors):
                            diffusion += self.volatility_function(t, T_maturity, k) * dW_correlated[k]
                        
                        # Update forward rate
                        if method == 'euler':
                            forward_rates[path, i, j] = forward_rates[path, i-1, j] + drift * dt_step + diffusion
                        elif method == 'milstein':
                            # Milstein scheme (more accurate for SDEs)
                            forward_rates[path, i, j] = forward_rates[path, i-1, j] + drift * dt_step + diffusion
                            # Add Milstein correction terms (simplified)
                            for k in range(self.n_factors):
                                vol = self.volatility_function(t, T_maturity, k)
                                forward_rates[path, i, j] += 0.5 * vol**2 * (dW_correlated[k]**2 - dt_step)
                        else:
                            raise ValueError("Method must be 'euler' or 'milstein'")
                    else:
                        # For past maturities, keep the last valid value
                        forward_rates[path, i, j] = forward_rates[path, i-1, j]
                
                # Calculate spot rate (shortest maturity forward rate)
                spot_rates[path, i] = forward_rates[path, i, 0]
        
        return times, forward_rates, spot_rates
    
    def bond_price(self, 
                   forward_rates: np.ndarray, 
                   t: float, 
                   T: float) -> float:
        """
        Calculate zero-coupon bond price from forward rates
        
        P(t,T) = exp(-∫ₜᵀ f(t,s)ds)
        
        Parameters:
        -----------
        forward_rates : np.ndarray
            Forward rate curve at time t
        t : float
            Current time
        T : float
            Bond maturity
            
        Returns:
        --------
        float
            Bond price
        """
        if T <= t:
            return 1.0
        
        # Find relevant maturities
        relevant_maturities = self.maturities[(self.maturities >= t) & (self.maturities <= T)]
        relevant_rates = forward_rates[(self.maturities >= t) & (self.maturities <= T)]
        
        if len(relevant_maturities) == 0:
            return 1.0
        
        # Integrate forward rates
        try:
            integral, _ = quad(lambda s: np.interp(s, relevant_maturities, relevant_rates), t, T)
            return np.exp(-integral)
        except (ValueError, RuntimeWarning):
            # Fallback to simple approximation
            if len(relevant_maturities) > 1:
                dt = relevant_maturities[1] - relevant_maturities[0]
                integral = np.sum(relevant_rates) * dt
                return np.exp(-integral)
            else:
                return np.exp(-relevant_rates[0] * (T - t))
    
    def yield_curve(self, 
                   forward_rates: np.ndarray, 
                   t: float) -> np.ndarray:
        """
        Calculate yield curve from forward rates
        
        Y(t,T) = (1/(T-t)) ∫ₜᵀ f(t,s)ds
        
        Parameters:
        -----------
        forward_rates : np.ndarray
            Forward rate curve at time t
        t : float
            Current time
            
        Returns:
        --------
        np.ndarray
            Yield curve
        """
        yields = np.zeros_like(self.maturities)
        
        for i, T in enumerate(self.maturities):
            if T > t:
                try:
                    integral, _ = quad(lambda s: np.interp(s, self.maturities, forward_rates), t, T)
                    yields[i] = integral / (T - t)
                except (ValueError, RuntimeWarning):
                    # Fallback to simple approximation
                    relevant_rates = forward_rates[self.maturities >= t]
                    relevant_maturities = self.maturities[self.maturities >= t]
                    if len(relevant_rates) > 0:
                        yields[i] = np.mean(relevant_rates[:min(len(relevant_rates), int((T-t)/0.25))])
            else:
                yields[i] = forward_rates[i]
        
        return yields
    
    def calibrate_to_market(self, 
                           market_yields: np.ndarray,
                           market_maturities: np.ndarray,
                           method: str = 'least_squares') -> Dict:
        """
        Calibrate model parameters to market data
        
        Parameters:
        -----------
        market_yields : np.ndarray
            Market yield curve
        market_maturities : np.ndarray
            Market maturity points
        method : str
            Calibration method ('least_squares' or 'maximum_likelihood')
            
        Returns:
        --------
        Dict
            Calibration results
        """
        if method == 'least_squares':
            return self._calibrate_least_squares(market_yields, market_maturities)
        elif method == 'maximum_likelihood':
            return self._calibrate_maximum_likelihood(market_yields, market_maturities)
        else:
            raise ValueError("Method must be 'least_squares' or 'maximum_likelihood'")
    
    def _calibrate_least_squares(self, 
                                market_yields: np.ndarray,
                                market_maturities: np.ndarray) -> Dict:
        """Least squares calibration"""
        # This is a simplified calibration - in practice, you'd need to
        # calibrate the volatility function parameters
        
        # Interpolate market yields to model maturities
        market_yields_interp = np.interp(self.maturities, market_maturities, market_yields)
        
        # Calculate model yields
        model_yields = self.yield_curve(self.initial_forward_curve, 0)
        
        # Calculate error
        error = np.sum((model_yields - market_yields_interp)**2)
        
        return {
            'error': error,
            'market_yields': market_yields_interp,
            'model_yields': model_yields,
            'calibrated': True
        }
    
    def _calibrate_maximum_likelihood(self, 
                                     market_yields: np.ndarray,
                                     market_maturities: np.ndarray) -> Dict:
        """Maximum likelihood calibration (placeholder)"""
        # This would involve fitting the model to historical data
        # and maximizing the likelihood function
        
        return {
            'error': 0.0,
            'calibrated': True,
            'method': 'maximum_likelihood'
        }
    
    def plot_forward_rate_surface(self, 
                                 times: np.ndarray, 
                                 forward_rates: np.ndarray,
                                 path_idx: int = 0,
                                 title: str = "HJM Forward Rate Surface"):
        """Plot forward rate surface"""
        fig = plt.figure(figsize=(15, 10))
        
        # 3D surface plot
        ax1 = fig.add_subplot(221, projection='3d')
        T_grid, t_grid = np.meshgrid(self.maturities, times)
        surface = forward_rates[path_idx, :, :]
        
        surf = ax1.plot_surface(T_grid, t_grid, surface, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Maturity (years)')
        ax1.set_ylabel('Time (years)')
        ax1.set_zlabel('Forward Rate')
        ax1.set_title('Forward Rate Surface')
        plt.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Contour plot
        ax2 = fig.add_subplot(222)
        contour = ax2.contour(T_grid, t_grid, surface, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('Maturity (years)')
        ax2.set_ylabel('Time (years)')
        ax2.set_title('Forward Rate Contours')
        
        # Forward rate curves at different times
        ax3 = fig.add_subplot(223)
        time_indices = np.linspace(0, len(times)-1, 10, dtype=int)
        for i in time_indices:
            ax3.plot(self.maturities, forward_rates[path_idx, i, :], 
                    label=f't={times[i]:.2f}', alpha=0.7)
        ax3.set_xlabel('Maturity (years)')
        ax3.set_ylabel('Forward Rate')
        ax3.set_title('Forward Rate Curves')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
        
        # Evolution of specific maturities
        ax4 = fig.add_subplot(224)
        maturity_indices = [0, 10, 30, 60, 90]  # Different maturities
        for j in maturity_indices:
            if j < len(self.maturities):
                ax4.plot(times, forward_rates[path_idx, :, j], 
                        label=f'T={self.maturities[j]:.1f}y', linewidth=2)
        ax4.set_xlabel('Time (years)')
        ax4.set_ylabel('Forward Rate')
        ax4.set_title('Forward Rate Evolution')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=16)
        plt.show()
    
    def plot_yield_curve_evolution(self, 
                                  times: np.ndarray, 
                                  forward_rates: np.ndarray,
                                  path_idx: int = 0,
                                  title: str = "Yield Curve Evolution"):
        """Plot yield curve evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Yield curves at different times
        ax1 = axes[0, 0]
        time_indices = np.linspace(0, len(times)-1, 8, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
        
        for i, color in zip(time_indices, colors):
            yields = self.yield_curve(forward_rates[path_idx, i, :], times[i])
            ax1.plot(self.maturities, yields, color=color, 
                    label=f't={times[i]:.2f}', linewidth=2)
        
        ax1.set_xlabel('Maturity (years)')
        ax1.set_ylabel('Yield')
        ax1.set_title('Yield Curve Evolution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # Yield curve shifts
        ax2 = axes[0, 1]
        initial_yields = self.yield_curve(forward_rates[path_idx, 0, :], times[0])
        final_yields = self.yield_curve(forward_rates[path_idx, -1, :], times[-1])
        
        ax2.plot(self.maturities, initial_yields, 'b-', label='Initial', linewidth=2)
        ax2.plot(self.maturities, final_yields, 'r-', label='Final', linewidth=2)
        ax2.fill_between(self.maturities, initial_yields, final_yields, alpha=0.3)
        ax2.set_xlabel('Maturity (years)')
        ax2.set_ylabel('Yield')
        ax2.set_title('Yield Curve Shift')
        ax2.legend()
        ax2.grid(True)
        
        # Term structure of volatility
        ax3 = axes[1, 0]
        yield_volatilities = np.std(forward_rates[path_idx, :, :], axis=0)
        ax3.plot(self.maturities, yield_volatilities, 'g-', linewidth=2)
        ax3.set_xlabel('Maturity (years)')
        ax3.set_ylabel('Volatility')
        ax3.set_title('Term Structure of Volatility')
        ax3.grid(True)
        
        # Yield curve statistics
        ax4 = axes[1, 1]
        all_yields = []
        for i in range(len(times)):
            yields = self.yield_curve(forward_rates[path_idx, i, :], times[i])
            all_yields.append(yields)
        
        all_yields = np.array(all_yields)
        mean_yields = np.mean(all_yields, axis=0)
        std_yields = np.std(all_yields, axis=0)
        
        ax4.plot(self.maturities, mean_yields, 'b-', label='Mean', linewidth=2)
        ax4.fill_between(self.maturities, mean_yields - std_yields, 
                        mean_yields + std_yields, alpha=0.3, label='±1σ')
        ax4.set_xlabel('Maturity (years)')
        ax4.set_ylabel('Yield')
        ax4.set_title('Yield Curve Statistics')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=16)
        plt.show()
    
    def analyze_model_properties(self, 
                                times: np.ndarray, 
                                forward_rates: np.ndarray,
                                spot_rates: np.ndarray) -> Dict:
        """
        Analyze model properties and statistics
        
        Parameters:
        -----------
        times : np.ndarray
            Time points
        forward_rates : np.ndarray
            Forward rate surfaces
        spot_rates : np.ndarray
            Spot rate paths
            
        Returns:
        --------
        Dict
            Analysis results
        """
        n_paths, n_times, n_maturities = forward_rates.shape
        
        # Calculate statistics
        analysis = {
            'n_paths': n_paths,
            'n_times': n_times,
            'n_maturities': n_maturities,
            'time_horizon': times[-1],
            'maturity_range': (self.maturities[0], self.maturities[-1])
        }
        
        # Spot rate statistics
        final_spot_rates = spot_rates[:, -1]
        analysis['spot_rate_stats'] = {
            'mean': np.mean(final_spot_rates),
            'std': np.std(final_spot_rates),
            'min': np.min(final_spot_rates),
            'max': np.max(final_spot_rates),
            'skewness': stats.skew(final_spot_rates),
            'kurtosis': stats.kurtosis(final_spot_rates)
        }
        
        # Forward rate statistics
        final_forward_rates = forward_rates[:, -1, :]
        analysis['forward_rate_stats'] = {
            'mean': np.mean(final_forward_rates, axis=0),
            'std': np.std(final_forward_rates, axis=0),
            'min': np.min(final_forward_rates, axis=0),
            'max': np.max(final_forward_rates, axis=0)
        }
        
        # Term structure analysis
        analysis['term_structure'] = {
            'slope': np.mean(np.diff(np.mean(final_forward_rates, axis=0))),
            'curvature': np.mean(np.diff(np.diff(np.mean(final_forward_rates, axis=0))))
        }
        
        # Volatility analysis
        forward_rate_changes = np.diff(forward_rates, axis=1)
        analysis['volatility'] = {
            'spot_rate_vol': np.std(np.diff(spot_rates, axis=1)),
            'forward_rate_vol': np.std(forward_rate_changes, axis=(0, 1))
        }
        
        return analysis
    
    def print_analysis(self, analysis: Dict):
        """Print model analysis results"""
        print("=" * 60)
        print("HJM MODEL ANALYSIS")
        print("=" * 60)
        
        print(f"Simulation Parameters:")
        print(f"  Paths: {analysis['n_paths']:,}")
        print(f"  Time steps: {analysis['n_times']:,}")
        print(f"  Maturities: {analysis['n_maturities']}")
        print(f"  Time horizon: {analysis['time_horizon']:.2f} years")
        print(f"  Maturity range: {analysis['maturity_range'][0]:.2f} - {analysis['maturity_range'][1]:.2f} years")
        
        print(f"\nSpot Rate Statistics:")
        stats = analysis['spot_rate_stats']
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Skewness: {stats['skewness']:.4f}")
        print(f"  Kurtosis: {stats['kurtosis']:.4f}")
        
        print(f"\nTerm Structure Analysis:")
        ts = analysis['term_structure']
        print(f"  Average slope: {ts['slope']:.6f}")
        print(f"  Average curvature: {ts['curvature']:.6f}")
        
        print(f"\nVolatility Analysis:")
        vol = analysis['volatility']
        print(f"  Spot rate volatility: {vol['spot_rate_vol']:.4f}")
        print(f"  Average forward rate volatility: {np.mean(vol['forward_rate_vol']):.4f}")
        
        print("=" * 60)


# Predefined volatility functions
def constant_volatility(sigma: float) -> Callable:
    """Constant volatility function"""
    return lambda t, T: sigma

def exponential_volatility(sigma: float, lambda_param: float) -> Callable:
    """Exponential volatility function: σ(t,T) = σ * exp(-λ(T-t))"""
    return lambda t, T: sigma * np.exp(-lambda_param * (T - t))

def hump_volatility(sigma: float, lambda_param: float, peak_time: float) -> Callable:
    """Hump-shaped volatility function"""
    return lambda t, T: sigma * (T - t) * np.exp(-lambda_param * (T - t - peak_time)**2)

def linear_volatility(sigma: float, slope: float) -> Callable:
    """Linear volatility function: σ(t,T) = σ * (1 + slope * (T-t))"""
    return lambda t, T: sigma * (1 + slope * (T - t))

def power_volatility(sigma: float, alpha: float) -> Callable:
    """Power volatility function: σ(t,T) = σ * (T-t)^α"""
    return lambda t, T: sigma * (T - t)**alpha


def create_sample_hjm_model(n_factors: int = 2, 
                           volatility_type: str = 'exponential') -> HJMForwardRateModel:
    """
    Create a sample HJM model with predefined parameters
    
    Parameters:
    -----------
    n_factors : int
        Number of factors
    volatility_type : str
        Type of volatility function ('constant', 'exponential', 'hump', 'linear', 'power')
        
    Returns:
    --------
    HJMForwardRateModel
        Configured HJM model
    """
    volatility_functions = []
    
    if volatility_type == 'constant':
        for i in range(n_factors):
            sigma = 0.01 + 0.005 * i  # Different volatilities for each factor
            volatility_functions.append(constant_volatility(sigma))
    
    elif volatility_type == 'exponential':
        for i in range(n_factors):
            sigma = 0.02 + 0.01 * i
            lambda_param = 0.1 + 0.05 * i
            volatility_functions.append(exponential_volatility(sigma, lambda_param))
    
    elif volatility_type == 'hump':
        for i in range(n_factors):
            sigma = 0.015 + 0.005 * i
            lambda_param = 0.2 + 0.1 * i
            peak_time = 2.0 + 1.0 * i
            volatility_functions.append(hump_volatility(sigma, lambda_param, peak_time))
    
    elif volatility_type == 'linear':
        for i in range(n_factors):
            sigma = 0.01 + 0.005 * i
            slope = 0.1 + 0.05 * i
            volatility_functions.append(linear_volatility(sigma, slope))
    
    elif volatility_type == 'power':
        for i in range(n_factors):
            sigma = 0.02 + 0.01 * i
            alpha = 0.5 + 0.2 * i
            volatility_functions.append(power_volatility(sigma, alpha))
    
    else:
        raise ValueError(f"Unknown volatility type: {volatility_type}")
    
    # Create correlation matrix (slightly correlated factors)
    correlation_matrix = np.eye(n_factors)
    if n_factors > 1:
        correlation_matrix[0, 1] = 0.3
        correlation_matrix[1, 0] = 0.3
    
    return HJMForwardRateModel(
        volatility_functions=volatility_functions,
        correlation_matrix=correlation_matrix
    )


if __name__ == "__main__":
    # Example usage
    print("HJM Model Example")
    print("=" * 50)
    
    # Create HJM model
    hjm_model = create_sample_hjm_model(n_factors=2, volatility_type='exponential')
    
    # Simulate forward rates
    print("Simulating forward rates...")
    times, forward_rates, spot_rates = hjm_model.simulate_forward_rates(
        T=5, dt=1/252, n_paths=100, method='euler'
    )
    
    # Analyze results
    analysis = hjm_model.analyze_model_properties(times, forward_rates, spot_rates)
    hjm_model.print_analysis(analysis)
    
    # Plot results
    print("Generating plots...")
    hjm_model.plot_forward_rate_surface(times, forward_rates, path_idx=0)
    hjm_model.plot_yield_curve_evolution(times, forward_rates, path_idx=0)
    
    # Calculate bond prices
    print("\nBond Pricing Example:")
    current_forward_curve = forward_rates[0, -1, :]  # Use final forward curve
    for maturity in [1, 2, 5, 10]:
        bond_price = hjm_model.bond_price(current_forward_curve, 0, maturity)
        print(f"  {maturity}-year bond price: {bond_price:.4f}")
    
    print("\nHJM model demonstration completed!")
