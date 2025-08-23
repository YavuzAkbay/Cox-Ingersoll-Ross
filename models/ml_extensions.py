"""
Machine Learning Extensions for Interest Rate Models

This module provides ML-based extensions to traditional interest rate models:
- Regime-switching models using Hidden Markov Models
- Stochastic volatility modeling
- Neural network-based yield curve forecasting
- Model comparison and validation
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns


class RegimeSwitchingModel:
    """
    Regime-switching interest rate model using Hidden Markov Models
    
    This model allows for different interest rate dynamics in different market regimes
    (e.g., high volatility, low volatility, crisis periods)
    """
    
    def __init__(self, n_regimes: int = 2, model_type: str = 'vasicek'):
        """
        Initialize regime-switching model
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes
        model_type : str
            Base model type ('vasicek' or 'cir')
        """
        self.n_regimes = n_regimes
        self.model_type = model_type
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.regime_models = []
        self.transition_matrix = None
        self.regime_probs = None
        
    def fit(self, rates: np.ndarray, dt: float = 1/252):
        """
        Fit regime-switching model to data
        
        Parameters:
        -----------
        rates : np.ndarray
            Interest rate time series
        dt : float
            Time step between observations
        """
        # Calculate features for regime detection
        features = self._extract_features(rates)
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit GMM to identify regimes
        self.gmm.fit(features_scaled)
        regime_labels = self.gmm.predict(features_scaled)
        
        # Estimate transition matrix
        self.transition_matrix = self._estimate_transition_matrix(regime_labels)
        
        # Fit individual models for each regime
        self.regime_models = []
        for regime in range(self.n_regimes):
            regime_mask = regime_labels == regime
            if np.sum(regime_mask) > 10:  # Need sufficient data
                regime_rates = rates[regime_mask]
                if self.model_type == 'vasicek':
                    from .vasicek_model import VasicekModel
                    model = VasicekModel(0.1, 0.05, 0.02)
                    kappa, theta, sigma = model.estimate_parameters(regime_rates, dt)
                    self.regime_models.append(VasicekModel(kappa, theta, sigma))
                elif self.model_type == 'cir':
                    from .cir_model import CIRModel
                    model = CIRModel(0.1, 0.05, 0.1)
                    kappa, theta, sigma = model.estimate_parameters(regime_rates, dt)
                    self.regime_models.append(CIRModel(kappa, theta, sigma))
        
        # Calculate regime probabilities
        self.regime_probs = self.gmm.predict_proba(features_scaled)
        
    def _extract_features(self, rates: np.ndarray) -> np.ndarray:
        """Extract features for regime detection"""
        features = []
        for i in range(len(rates)):
            if i < 20:  # Need enough history
                features.append([rates[i], 0, 0, 0])
            else:
                # Current rate, volatility, trend, mean reversion
                window = rates[max(0, i-20):i+1]
                volatility = np.std(np.diff(window))
                trend = (window[-1] - window[0]) / len(window)
                mean_reversion = np.mean(window) - window[-1]
                
                features.append([rates[i], volatility, trend, mean_reversion])
        
        return np.array(features)
    
    def _estimate_transition_matrix(self, regime_labels: np.ndarray) -> np.ndarray:
        """Estimate transition matrix between regimes"""
        n_regimes = self.n_regimes
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            current_regime = regime_labels[i]
            next_regime = regime_labels[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    def simulate(self, T: float, dt: float = 1/252, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate regime-switching process
        
        Returns:
        --------
        times : np.ndarray
            Time points
        rates : np.ndarray
            Simulated interest rates
        regimes : np.ndarray
            Regime labels
        """
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        
        rates = np.zeros((n_paths, n_steps + 1))
        regimes = np.zeros((n_paths, n_steps + 1), dtype=int)
        
        for path in range(n_paths):
            # Start with random regime
            current_regime = np.random.choice(self.n_regimes)
            current_rate = self.regime_models[current_regime].r0
            
            rates[path, 0] = current_rate
            regimes[path, 0] = current_regime
            
            for i in range(1, n_steps + 1):
                # Simulate regime transition
                current_regime = np.random.choice(
                    self.n_regimes, 
                    p=self.transition_matrix[current_regime, :]
                )
                
                # Simulate rate evolution in current regime
                model = self.regime_models[current_regime]
                _, rate_path = model.simulate_path(dt, dt, n_paths=1, method='exact')
                
                current_rate = rate_path[0, -1]
                
                rates[path, i] = current_rate
                regimes[path, i] = current_regime
        
        return times, rates, regimes
    
    def plot_regimes(self, rates: np.ndarray, regime_probs: np.ndarray = None):
        """Plot regime analysis"""
        plt.figure(figsize=(15, 10))
        
        # Plot interest rates with regime colors
        plt.subplot(2, 2, 1)
        if regime_probs is not None:
            dominant_regime = np.argmax(regime_probs, axis=1)
            colors = plt.cm.Set1(dominant_regime / self.n_regimes)
            plt.scatter(range(len(rates)), rates, c=colors, alpha=0.6)
        else:
            plt.plot(rates)
        plt.xlabel('Time')
        plt.ylabel('Interest Rate')
        plt.title('Interest Rates by Regime')
        plt.grid(True)
        
        # Plot regime probabilities
        if regime_probs is not None:
            plt.subplot(2, 2, 2)
            for regime in range(self.n_regimes):
                plt.plot(regime_probs[:, regime], label=f'Regime {regime}')
            plt.xlabel('Time')
            plt.ylabel('Regime Probability')
            plt.title('Regime Probabilities')
            plt.legend()
            plt.grid(True)
        
        # Plot transition matrix
        if self.transition_matrix is not None:
            plt.subplot(2, 2, 3)
            sns.heatmap(self.transition_matrix, annot=True, cmap='Blues', 
                       xticklabels=[f'R{i}' for i in range(self.n_regimes)],
                       yticklabels=[f'R{i}' for i in range(self.n_regimes)])
            plt.title('Regime Transition Matrix')
        
        # Plot regime characteristics
        plt.subplot(2, 2, 4)
        if self.regime_models:
            for i, model in enumerate(self.regime_models):
                if hasattr(model, 'theta'):
                    plt.bar(f'Regime {i}', model.theta, label=f'θ={model.theta:.3f}')
            plt.ylabel('Long-term Mean (θ)')
            plt.title('Regime Characteristics')
        
        plt.tight_layout()
        plt.show()


class StochasticVolatilityModel:
    """
    Stochastic volatility extension for interest rate models
    
    This model allows for time-varying volatility in interest rate dynamics
    """
    
    def __init__(self, base_model_type: str = 'vasicek'):
        """
        Initialize stochastic volatility model
        
        Parameters:
        -----------
        base_model_type : str
            Base interest rate model ('vasicek' or 'cir')
        """
        self.base_model_type = base_model_type
        self.kappa_r = None  # Mean reversion for rates
        self.theta_r = None  # Long-term mean for rates
        self.kappa_v = None  # Mean reversion for volatility
        self.theta_v = None  # Long-term mean for volatility
        self.sigma_v = None  # Volatility of volatility
        self.rho = None      # Correlation between rate and volatility shocks
        
    def fit(self, rates: np.ndarray, dt: float = 1/252):
        """
        Fit stochastic volatility model to data
        
        Parameters:
        -----------
        rates : np.ndarray
            Interest rate time series
        dt : float
            Time step between observations
        """
        # Estimate volatility process
        returns = np.diff(rates)
        volatility = np.abs(returns)
        
        # Fit volatility process (assuming mean-reverting)
        self.kappa_v = 0.1  # Speed of mean reversion for volatility
        self.theta_v = np.mean(volatility)
        self.sigma_v = np.std(volatility)
        
        # Fit interest rate process
        self.kappa_r = 0.1
        self.theta_r = np.mean(rates)
        
        # Estimate correlation
        self.rho = np.corrcoef(returns[:-1], volatility[1:])[0, 1]
        
    def simulate(self, T: float, dt: float = 1/252, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate stochastic volatility process
        
        Returns:
        --------
        times : np.ndarray
            Time points
        rates : np.ndarray
            Simulated interest rates
        volatilities : np.ndarray
            Simulated volatilities
        """
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        
        rates = np.zeros((n_paths, n_steps + 1))
        volatilities = np.zeros((n_paths, n_steps + 1))
        
        for path in range(n_paths):
            # Initialize
            rates[path, 0] = self.theta_r
            volatilities[path, 0] = self.theta_v
            
            for i in range(1, n_steps + 1):
                # Generate correlated shocks
                z1 = np.random.normal(0, 1)
                z2 = np.random.normal(0, 1)
                z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
                
                # Update volatility
                volatilities[path, i] = volatilities[path, i-1] + \
                    self.kappa_v * (self.theta_v - volatilities[path, i-1]) * dt + \
                    self.sigma_v * np.sqrt(volatilities[path, i-1]) * z2 * np.sqrt(dt)
                volatilities[path, i] = np.maximum(volatilities[path, i], 0.001)
                
                # Update interest rate
                if self.base_model_type == 'vasicek':
                    rates[path, i] = rates[path, i-1] + \
                        self.kappa_r * (self.theta_r - rates[path, i-1]) * dt + \
                        volatilities[path, i] * z1 * np.sqrt(dt)
                else:  # CIR
                    rates[path, i] = rates[path, i-1] + \
                        self.kappa_r * (self.theta_r - rates[path, i-1]) * dt + \
                        volatilities[path, i] * np.sqrt(np.maximum(rates[path, i-1], 0)) * z1 * np.sqrt(dt)
                    rates[path, i] = np.maximum(rates[path, i], 0)
        
        return times, rates, volatilities
    
    def plot_simulation(self, times: np.ndarray, rates: np.ndarray, volatilities: np.ndarray):
        """Plot stochastic volatility simulation"""
        plt.figure(figsize=(15, 10))
        
        # Plot interest rates
        plt.subplot(2, 3, 1)
        for i in range(min(5, rates.shape[0])):
            plt.plot(times, rates[i, :], alpha=0.7)
        plt.xlabel('Time (years)')
        plt.ylabel('Interest Rate')
        plt.title('Interest Rate Paths')
        plt.grid(True)
        
        # Plot volatilities
        plt.subplot(2, 3, 2)
        for i in range(min(5, volatilities.shape[0])):
            plt.plot(times, volatilities[i, :], alpha=0.7)
        plt.xlabel('Time (years)')
        plt.ylabel('Volatility')
        plt.title('Volatility Paths')
        plt.grid(True)
        
        # Plot rate vs volatility
        plt.subplot(2, 3, 3)
        plt.scatter(rates.flatten(), volatilities.flatten(), alpha=0.1)
        plt.xlabel('Interest Rate')
        plt.ylabel('Volatility')
        plt.title('Rate vs Volatility')
        plt.grid(True)
        
        # Plot volatility distribution
        plt.subplot(2, 3, 4)
        plt.hist(volatilities.flatten(), bins=50, density=True, alpha=0.7)
        plt.xlabel('Volatility')
        plt.ylabel('Density')
        plt.title('Volatility Distribution')
        plt.grid(True)
        
        # Plot rate distribution
        plt.subplot(2, 3, 5)
        plt.hist(rates.flatten(), bins=50, density=True, alpha=0.7)
        plt.xlabel('Interest Rate')
        plt.ylabel('Density')
        plt.title('Rate Distribution')
        plt.grid(True)
        
        # Plot correlation
        plt.subplot(2, 3, 6)
        plt.scatter(np.diff(rates[0, :]), volatilities[0, 1:], alpha=0.5)
        plt.xlabel('Rate Changes')
        plt.ylabel('Volatility')
        plt.title('Rate Changes vs Volatility')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class NeuralNetworkForecaster:
    """
    Neural network-based yield curve forecasting
    
    This model uses deep learning to predict future yield curves
    """
    
    def __init__(self, sequence_length: int = 20, hidden_size: int = 64):
        """
        Initialize neural network forecaster
        
        Parameters:
        -----------
        sequence_length : int
            Number of past observations to use for prediction
        hidden_size : int
            Size of hidden layers
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, rates: np.ndarray, maturities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for neural network training
        
        Parameters:
        -----------
        rates : np.ndarray
            Interest rate time series
        maturities : np.ndarray
            Yield curve maturities
            
        Returns:
        --------
        X : np.ndarray
            Input sequences
        y : np.ndarray
            Target values
        """
        # Calculate yield curves for each time point
        from .vasicek_model import VasicekModel
        model = VasicekModel(0.1, 0.05, 0.02)
        
        yield_curves = []
        for rate in rates:
            yields = model.yield_curve(rate, maturities)
            yield_curves.append(yields)
        
        yield_curves = np.array(yield_curves)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(yield_curves)):
            X.append(yield_curves[i-self.sequence_length:i])
            y.append(yield_curves[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build neural network model"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            model = keras.Sequential([
                layers.LSTM(self.hidden_size, return_sequences=True, input_shape=input_shape),
                layers.Dropout(0.2),
                layers.LSTM(self.hidden_size // 2),
                layers.Dropout(0.2),
                layers.Dense(input_shape[1])
            ])
            
            model.compile(optimizer='adam', loss='mse')
            self.model = model
            
        except ImportError:
            print("TensorFlow not available. Using simple linear model.")
            self.model = SimpleLinearModel(input_shape)
    
    def fit(self, rates: np.ndarray, maturities: np.ndarray, epochs: int = 100):
        """
        Fit neural network model
        
        Parameters:
        -----------
        rates : np.ndarray
            Interest rate time series
        maturities : np.ndarray
            Yield curve maturities
        epochs : int
            Number of training epochs
        """
        X, y = self.prepare_data(rates, maturities)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        self.build_model(X.shape[1:])
        
        # Train model
        if hasattr(self.model, 'fit'):
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
            return history
        else:
            self.model.fit(X_train, y_train)
    
    def predict(self, rates: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        """
        Predict future yield curves
        
        Parameters:
        -----------
        rates : np.ndarray
            Recent interest rate history
        maturities : np.ndarray
            Yield curve maturities
            
        Returns:
        --------
        np.ndarray
            Predicted yield curves
        """
        X, _ = self.prepare_data(rates, maturities)
        if len(X) > 0:
            return self.model.predict(X)
        else:
            return np.array([])


class SimpleLinearModel:
    """Simple linear model as fallback when TensorFlow is not available"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.weights = None
        
    def fit(self, X, y):
        # Reshape for linear regression
        X_flat = X.reshape(X.shape[0], -1)
        self.weights = np.linalg.lstsq(X_flat, y, rcond=None)[0]
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return X_flat @ self.weights


def compare_models(models: Dict, rates: np.ndarray, dt: float = 1/252):
    """
    Compare different interest rate models
    
    Parameters:
    -----------
    models : Dict
        Dictionary of model instances
    rates : np.ndarray
        Observed interest rates
    dt : float
        Time step
    """
    results = {}
    
    for name, model in models.items():
        try:
            # Fit model
            if hasattr(model, 'fit'):
                model.fit(rates, dt)
            
            # Simulate
            if hasattr(model, 'simulate'):
                times, sim_rates, *_ = model.simulate(1, dt, n_paths=1000)
                
                # Calculate metrics
                mean_rate = np.mean(sim_rates[:, -1])
                std_rate = np.std(sim_rates[:, -1])
                
                results[name] = {
                    'mean': mean_rate,
                    'std': std_rate,
                    'sim_rates': sim_rates
                }
        except Exception as e:
            print(f"Error with model {name}: {e}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Compare distributions
    plt.subplot(2, 3, 1)
    for name, result in results.items():
        plt.hist(result['sim_rates'][:, -1], bins=30, alpha=0.7, label=name)
    plt.xlabel('Interest Rate')
    plt.ylabel('Density')
    plt.title('Final Rate Distributions')
    plt.legend()
    plt.grid(True)
    
    # Compare means
    plt.subplot(2, 3, 2)
    names = list(results.keys())
    means = [results[name]['mean'] for name in names]
    plt.bar(names, means)
    plt.ylabel('Mean Rate')
    plt.title('Model Means')
    plt.grid(True)
    
    # Compare volatilities
    plt.subplot(2, 3, 3)
    stds = [results[name]['std'] for name in names]
    plt.bar(names, stds)
    plt.ylabel('Standard Deviation')
    plt.title('Model Volatilities')
    plt.grid(True)
    
    # Compare paths
    plt.subplot(2, 3, 4)
    times = np.linspace(0, 1, results[names[0]]['sim_rates'].shape[1])
    for name in names[:3]:  # Plot first 3 models
        mean_path = np.mean(results[name]['sim_rates'], axis=0)
        plt.plot(times, mean_path, label=name)
    plt.xlabel('Time (years)')
    plt.ylabel('Interest Rate')
    plt.title('Mean Paths')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Example usage
    print("ML Extensions Example")
    print("=" * 50)
    
    # Create sample data
    from .vasicek_model import create_sample_data
    rates = create_sample_data(n_obs=1000)
    
    # Test regime-switching model
    print("Testing regime-switching model...")
    regime_model = RegimeSwitchingModel(n_regimes=2, model_type='vasicek')
    regime_model.fit(rates)
    
    # Test stochastic volatility model
    print("Testing stochastic volatility model...")
    sv_model = StochasticVolatilityModel(base_model_type='vasicek')
    sv_model.fit(rates)
    times, sim_rates, sim_vols = sv_model.simulate(1, n_paths=100)
    sv_model.plot_simulation(times, sim_rates, sim_vols)
    
    # Test neural network forecaster
    print("Testing neural network forecaster...")
    maturities = np.array([0.5, 1, 2, 5, 10])
    nn_model = NeuralNetworkForecaster(sequence_length=20)
    nn_model.fit(rates, maturities, epochs=50)
    
    # Compare models
    print("Comparing models...")
    models = {
        'Vasicek': sv_model,
        'Regime-Switching': regime_model
    }
    compare_models(models, rates)
