"""
Utility Helper Functions for Interest Rate Models

Simple utility functions for data analysis, risk metrics, and model validation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
import matplotlib.pyplot as plt


def calculate_risk_metrics(returns: np.ndarray, confidence_level: float = 0.95) -> Dict:
    """Calculate comprehensive risk metrics"""
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = np.mean(returns)
    metrics['std'] = np.std(returns)
    metrics['skewness'] = stats.skew(returns)
    metrics['kurtosis'] = stats.kurtosis(returns)
    
    # Percentiles
    metrics['min'] = np.min(returns)
    metrics['max'] = np.max(returns)
    metrics['median'] = np.median(returns)
    
    # VaR and CVaR
    alpha = 1 - confidence_level
    var_percentile = alpha * 100
    cvar_percentile = alpha * 100
    
    metrics['var'] = np.percentile(returns, var_percentile)
    metrics['cvar'] = np.mean(returns[returns <= metrics['var']])
    
    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = np.min(drawdown)
    
    # Volatility metrics
    metrics['annualized_vol'] = metrics['std'] * np.sqrt(252)
    metrics['vol_of_vol'] = np.std(np.abs(returns))
    
    return metrics


def calculate_yield_curve_metrics(yields: np.ndarray, maturities: np.ndarray) -> Dict:
    """Calculate yield curve metrics"""
    metrics = {}
    
    # Level, slope, and curvature
    if len(yields) >= 3:
        metrics['level'] = np.mean(yields)
        metrics['slope'] = yields[-1] - yields[0]
        
        if len(yields) % 2 == 0:
            mid_idx = len(yields) // 2
        else:
            mid_idx = len(yields) // 2
        metrics['curvature'] = yields[mid_idx] - (yields[0] + yields[-1]) / 2
    
    # Duration and convexity (approximate)
    if len(yields) > 1:
        duration = np.sum(maturities * np.exp(-yields * maturities)) / np.sum(np.exp(-yields * maturities))
        metrics['duration'] = duration
        
        convexity = np.sum(maturities**2 * np.exp(-yields * maturities)) / np.sum(np.exp(-yields * maturities))
        metrics['convexity'] = convexity
    
    return metrics


def test_stationarity(series: np.ndarray, test_type: str = 'adf') -> Dict:
    """Test for stationarity of time series"""
    from statsmodels.tsa.stattools import adfuller, kpss
    
    results = {}
    
    if test_type == 'adf':
        adf_result = adfuller(series)
        results['test_statistic'] = adf_result[0]
        results['p_value'] = adf_result[1]
        results['critical_values'] = adf_result[4]
        results['is_stationary'] = adf_result[1] < 0.05
        
    elif test_type == 'kpss':
        kpss_result = kpss(series)
        results['test_statistic'] = kpss_result[0]
        results['p_value'] = kpss_result[1]
        results['critical_values'] = kpss_result[3]
        results['is_stationary'] = kpss_result[1] > 0.05
        
    return results


def calculate_autocorrelation(series: np.ndarray, max_lag: int = 20) -> tuple:
    """Calculate autocorrelation function"""
    lags = np.arange(1, max_lag + 1)
    autocorr = []
    
    for lag in lags:
        if lag < len(series):
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            autocorr.append(corr)
        else:
            autocorr.append(np.nan)
    
    return lags, np.array(autocorr)


def plot_autocorrelation(series: np.ndarray, max_lag: int = 20, title: str = "Autocorrelation Function"):
    """Plot autocorrelation function"""
    lags, autocorr = calculate_autocorrelation(series, max_lag)
    
    plt.figure(figsize=(10, 6))
    plt.bar(lags, autocorr, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1.96/np.sqrt(len(series)), color='red', linestyle='--', alpha=0.7, label='95% CI')
    plt.axhline(y=-1.96/np.sqrt(len(series)), color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_rolling_statistics(series: np.ndarray, window: int = 30) -> Dict:
    """Calculate rolling statistics"""
    stats = {}
    
    stats['rolling_mean'] = pd.Series(series).rolling(window=window).mean().values
    stats['rolling_std'] = pd.Series(series).rolling(window=window).std().values
    stats['rolling_skew'] = pd.Series(series).rolling(window=window).skew().values
    stats['rolling_kurt'] = pd.Series(series).rolling(window=window).kurt().values
    
    return stats


def plot_rolling_statistics(series: np.ndarray, window: int = 30, title: str = "Rolling Statistics"):
    """Plot rolling statistics"""
    stats = calculate_rolling_statistics(series, window)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original series
    axes[0, 0].plot(series, label='Original', alpha=0.7)
    axes[0, 0].plot(stats['rolling_mean'], label=f'Rolling Mean ({window})', linewidth=2)
    axes[0, 0].set_title('Time Series with Rolling Mean')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rolling standard deviation
    axes[0, 1].plot(stats['rolling_std'], label=f'Rolling Std ({window})', color='red')
    axes[0, 1].set_title('Rolling Standard Deviation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rolling skewness
    axes[1, 0].plot(stats['rolling_skew'], label=f'Rolling Skewness ({window})', color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Rolling Skewness')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rolling kurtosis
    axes[1, 1].plot(stats['rolling_kurt'], label=f'Rolling Kurtosis ({window})', color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Rolling Kurtosis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()


def calculate_model_performance_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """Calculate model performance metrics"""
    metrics = {}
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return metrics
    
    # Mean squared error
    metrics['mse'] = np.mean((actual_clean - predicted_clean)**2)
    
    # Root mean squared error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean absolute error
    metrics['mae'] = np.mean(np.abs(actual_clean - predicted_clean))
    
    # Mean absolute percentage error
    metrics['mape'] = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
    
    # R-squared
    ss_res = np.sum((actual_clean - predicted_clean)**2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean))**2)
    metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Correlation
    metrics['correlation'] = np.corrcoef(actual_clean, predicted_clean)[0, 1]
    
    return metrics


def plot_model_performance(actual: np.ndarray, predicted: np.ndarray, title: str = "Model Performance"):
    """Plot model performance analysis"""
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        print("No valid data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scatter plot
    axes[0, 0].scatter(actual_clean, predicted_clean, alpha=0.6)
    min_val = min(np.min(actual_clean), np.min(predicted_clean))
    max_val = max(np.max(actual_clean), np.max(predicted_clean))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = actual_clean - predicted_clean
    axes[0, 1].scatter(predicted_clean, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time series comparison
    axes[1, 0].plot(actual_clean, label='Actual', alpha=0.7)
    axes[1, 0].plot(predicted_clean, label='Predicted', alpha=0.7)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Time Series Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', label='Mean')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
    
    # Print metrics
    metrics = calculate_model_performance_metrics(actual, predicted)
    print(f"\nModel Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def generate_sample_data(n_obs: int, model_type: str = 'vasicek', **kwargs) -> np.ndarray:
    """Generate sample data for testing"""
    if model_type == 'vasicek':
        from models.vasicek_model import VasicekModel
        kappa = kwargs.get('kappa', 0.1)
        theta = kwargs.get('theta', 0.05)
        sigma = kwargs.get('sigma', 0.02)
        r0 = kwargs.get('r0', 0.05)
        
        model = VasicekModel(kappa, theta, sigma, r0)
        return model.create_sample_data(n_obs, **kwargs)
        
    elif model_type == 'cir':
        from models.cir_model import CIRModel
        kappa = kwargs.get('kappa', 0.1)
        theta = kwargs.get('theta', 0.05)
        sigma = kwargs.get('sigma', 0.1)
        r0 = kwargs.get('r0', 0.05)
        
        model = CIRModel(kappa, theta, sigma, r0)
        return model.create_sample_data(n_obs, **kwargs)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_summary_statistics(data: np.ndarray, title: str = "Summary Statistics"):
    """Print comprehensive summary statistics"""
    print(f"\n{title}")
    print("=" * 50)
    
    # Basic statistics
    print(f"Count: {len(data)}")
    print(f"Mean: {np.mean(data):.6f}")
    print(f"Std: {np.std(data):.6f}")
    print(f"Min: {np.min(data):.6f}")
    print(f"25%: {np.percentile(data, 25):.6f}")
    print(f"50%: {np.median(data):.6f}")
    print(f"75%: {np.percentile(data, 75):.6f}")
    print(f"Max: {np.max(data):.6f}")
    
    # Additional statistics
    print(f"Skewness: {stats.skew(data):.6f}")
    print(f"Kurtosis: {stats.kurtosis(data):.6f}")
    
    # Risk metrics
    risk_metrics = calculate_risk_metrics(data)
    print(f"VaR (95%): {risk_metrics['var']:.6f}")
    print(f"CVaR (95%): {risk_metrics['cvar']:.6f}")
    print(f"Max Drawdown: {risk_metrics['max_drawdown']:.6f}")


if __name__ == "__main__":
    # Example usage
    print("Utility Functions Example")
    print("=" * 50)
    
    # Generate sample data
    sample_data = generate_sample_data(1000, 'vasicek')
    
    # Print summary statistics
    print_summary_statistics(sample_data, "Sample Interest Rate Data")
    
    # Calculate and print risk metrics
    risk_metrics = calculate_risk_metrics(sample_data)
    print(f"\nRisk Metrics:")
    for key, value in risk_metrics.items():
        print(f"{key}: {value:.6f}")
    
    # Plot autocorrelation
    plot_autocorrelation(sample_data, title="Interest Rate Autocorrelation")
    
    # Plot rolling statistics
    plot_rolling_statistics(sample_data, title="Interest Rate Rolling Statistics")
