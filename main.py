"""Cox-Ingersoll-Ross and Vasicek Interest Rate Models"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import CIRModel, VasicekModel, RegimeSwitchingModel, StochasticVolatilityModel
from data import InterestRateDataFetcher
from utils import calculate_risk_metrics, print_summary_statistics


def run_demo():
    
    print("=" * 60)
    print("Cox-Ingersoll-Ross & Vasicek Interest Rate Models")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("1. BASIC MODEL DEMONSTRATION")
    print("-" * 30)
    
    cir_model = CIRModel(kappa=0.15, theta=0.055, sigma=0.08, r0=0.045)
    vasicek_model = VasicekModel(kappa=0.15, theta=0.055, sigma=0.015, r0=0.045)
    
    print(f"CIR Model: κ={0.15}, θ={0.055}, σ={0.08}")
    print(f"Vasicek Model: κ={0.15}, θ={0.055}, σ={0.015}")
    
    T, dt, n_paths = 30, 1/252, 1000
    print(f"\nSimulating {n_paths} paths for {T} years...")
    
    cir_times, cir_rates = cir_model.simulate_path(T, dt, n_paths, method='exact')
    vasicek_times, vasicek_rates = vasicek_model.simulate_path(T, dt, n_paths, method='exact')
    
    cir_final = cir_rates[:, -1]
    vasicek_final = vasicek_rates[:, -1]
    
    print("\nFinal Rate Statistics:")
    print(f"{'Model':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    print(f"{'CIR':<12} {np.mean(cir_final):<10.4f} {np.std(cir_final):<10.4f} {np.min(cir_final):<10.4f} {np.max(cir_final):<10.4f}")
    print(f"{'Vasicek':<12} {np.mean(vasicek_final):<10.4f} {np.std(vasicek_final):<10.4f} {np.min(vasicek_final):<10.4f} {np.max(vasicek_final):<10.4f}")
    
    print("\n\n2. BOND PRICING AND YIELD CURVES")
    print("-" * 30)
    
    current_rate = 0.045
    maturities = np.array([0.5, 1, 2, 5, 10, 20, 30])
    
    cir_prices = [cir_model.bond_price(current_rate, T) for T in maturities]
    vasicek_prices = [vasicek_model.bond_price(current_rate, T) for T in maturities]
    cir_yields = cir_model.yield_curve(current_rate, maturities)
    vasicek_yields = vasicek_model.yield_curve(current_rate, maturities)
    
    print(f"Bond prices and yields for current rate = {current_rate:.1%}")
    print(f"{'Maturity':<10} {'CIR Price':<12} {'Vasicek Price':<12} {'CIR Yield':<12} {'Vasicek Yield':<12}")
    print("-" * 65)
    for i, T in enumerate(maturities):
        print(f"{T:<10.1f} {cir_prices[i]:<12.4f} {vasicek_prices[i]:<12.4f} {cir_yields[i]:<12.4f} {vasicek_yields[i]:<12.4f}")
    
    print("\n\n3. DATA ANALYSIS")
    print("-" * 30)
    
    fetcher = InterestRateDataFetcher()
    print("Fetching sample interest rate data...")
    yield_df = fetcher.fetch_treasury_yields(start_date='2020-01-01')
    
    if not yield_df.empty:
        short_rate = yield_df['10Y'].dropna()
        rates_array = short_rate.values
        
        print(f"Fetched {len(rates_array)} days of 10-year Treasury yield data")
        print(f"Current yield: {short_rate.iloc[-1]:.4f}")
        
        print("\nFitting models to real data...")
        cir_kappa, cir_theta, cir_sigma = cir_model.estimate_parameters(rates_array, 1/252)
        vasicek_kappa, vasicek_theta, vasicek_sigma = vasicek_model.estimate_parameters(rates_array, 1/252)
        
        print("Estimated Parameters:")
        print(f"CIR: κ={cir_kappa:.4f}, θ={cir_theta:.4f}, σ={cir_sigma:.4f}")
        print(f"Vasicek: κ={vasicek_kappa:.4f}, θ={vasicek_theta:.4f}, σ={vasicek_sigma:.4f}")
        
        print("\nRisk Metrics for Historical Data:")
        risk_metrics = calculate_risk_metrics(rates_array)
        for key, value in risk_metrics.items():
            print(f"{key}: {value:.6f}")
    
    print("\n\n4. MACHINE LEARNING EXTENSIONS")
    print("-" * 30)
    
    print("Testing regime-switching model...")
    np.random.seed(42)
    n_obs = 1000
    dt = 1/252
    
    regime1_model = VasicekModel(0.15, 0.045, 0.015, r0=0.04)
    times1, rates1 = regime1_model.simulate_path(n_obs//2 * dt, dt, n_paths=1)
    
    regime2_model = VasicekModel(0.15, 0.065, 0.025, r0=0.06)
    times2, rates2 = regime2_model.simulate_path(n_obs//2 * dt, dt, n_paths=1)
    
    rates_ml = np.concatenate([rates1[0, :], rates2[0, :]])
    
    regime_model = RegimeSwitchingModel(n_regimes=2, model_type='vasicek')
    regime_model.fit(rates_ml, dt)
    print("Regime-switching model fitted successfully!")
    
    print("\nTesting stochastic volatility model...")
    sv_model = StochasticVolatilityModel(base_model_type='vasicek')
    sv_model.fit(rates_ml, dt)
    print("Stochastic volatility model fitted successfully!")
    print(f"Fitted parameters: κ_r={sv_model.kappa_r:.4f}, θ_r={sv_model.theta_r:.4f}, σ_v={sv_model.sigma_v:.4f}")
    
    print("\n\n5. MODEL COMPARISON")
    print("-" * 30)
    
    models = {
        'CIR': cir_model,
        'Vasicek': vasicek_model,
        'Regime-Switching': regime_model,
        'Stochastic Volatility': sv_model
    }
    
    print("Comparing model simulations...")
    results = {}
    
    for name, model in models.items():
        try:
            if hasattr(model, 'simulate'):
                times, sim_rates, *_ = model.simulate(1, dt, n_paths=100)
                mean_rate = np.mean(sim_rates[:, -1])
                std_rate = np.std(sim_rates[:, -1])
                results[name] = {'mean': mean_rate, 'std': std_rate}
            else:
                times, sim_rates = model.simulate_path(1, dt, n_paths=100)
                mean_rate = np.mean(sim_rates[:, -1])
                std_rate = np.std(sim_rates[:, -1])
                results[name] = {'mean': mean_rate, 'std': std_rate}
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    print("\nModel Comparison Results:")
    print(f"{'Model':<20} {'Mean':<10} {'Std':<10}")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:<20} {result['mean']:<10.4f} {result['std']:<10.4f}")
    
    print("\n\n6. VISUALIZATION")
    print("-" * 30)
    
    print("Generating plots...")
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    plt.subplot(3, 3, 1)
    max_time_idx = min(len(cir_times), int(10 / dt))
    
    for i in range(min(5, cir_rates.shape[0])):
        plt.plot(cir_times[:max_time_idx], cir_rates[i, :max_time_idx], 'b-', alpha=0.7, label='CIR' if i == 0 else "")
        plt.plot(vasicek_times[:max_time_idx], vasicek_rates[i, :max_time_idx], 'r-', alpha=0.7, label='Vasicek' if i == 0 else "")
    plt.xlabel('Time (years)')
    plt.ylabel('Interest Rate')
    plt.title('Interest Rate Paths Comparison (First 10 Years)')
    plt.legend()
    plt.grid(True)
    
    # Yield curves
    plt.subplot(3, 3, 2)
    maturities_fine = np.linspace(0.1, 30, 100)
    cir_yields_fine = cir_model.yield_curve(current_rate, maturities_fine)
    vasicek_yields_fine = vasicek_model.yield_curve(current_rate, maturities_fine)
    plt.plot(maturities_fine, cir_yields_fine, 'b-', label='CIR', linewidth=2)
    plt.plot(maturities_fine, vasicek_yields_fine, 'r-', label='Vasicek', linewidth=2)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.title(f'Yield Curves (r={current_rate:.1%})')
    plt.legend()
    plt.grid(True)
    
    # Rate distributions
    plt.subplot(3, 3, 3)
    plt.hist(cir_final, bins=30, alpha=0.7, label='CIR', density=True)
    plt.hist(vasicek_final, bins=30, alpha=0.7, label='Vasicek', density=True)
    plt.xlabel('Interest Rate')
    plt.ylabel('Density')
    plt.title('Final Rate Distributions')
    plt.legend()
    plt.grid(True)
    
    # Bond prices
    plt.subplot(3, 3, 4)
    plt.plot(maturities, cir_prices, 'bo-', label='CIR', linewidth=2, markersize=8)
    plt.plot(maturities, vasicek_prices, 'ro-', label='Vasicek', linewidth=2, markersize=8)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Bond Price')
    plt.title('Bond Prices')
    plt.legend()
    plt.grid(True)
    
    # Historical data (if available)
    if not yield_df.empty:
        plt.subplot(3, 3, 5)
        plt.plot(short_rate.index, short_rate.values, 'g-', linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('10-Year Treasury Yield')
        plt.title('Historical Interest Rates')
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Rolling volatility
        plt.subplot(3, 3, 6)
        rate_changes = np.diff(rates_array)
        rolling_vol = pd.Series(rate_changes).rolling(30).std()
        plt.plot(rolling_vol.index, rolling_vol.values, 'b-', label='Rolling Volatility')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.title('Time-Varying Volatility')
        plt.legend()
        plt.grid(True)
    
    # Regime analysis
    plt.subplot(3, 3, 7)
    if regime_model.regime_probs is not None:
        for regime in range(regime_model.n_regimes):
            plt.plot(regime_model.regime_probs[:, regime], label=f'Regime {regime}')
        plt.xlabel('Time')
        plt.ylabel('Regime Probability')
        plt.title('Regime Probabilities')
        plt.legend()
        plt.grid(True)
    
    # Model comparison
    plt.subplot(3, 3, 8)
    model_names = list(results.keys())
    means = [results[name]['mean'] for name in model_names]
    stds = [results[name]['std'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', alpha=0.7)
    plt.bar(x + width/2, stds, width, label='Std Dev', alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Model Statistics Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Summary statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
    Project Summary:
    
    Models Implemented:
    • CIR (Cox-Ingersoll-Ross)
    • Vasicek
    • Regime-Switching
    • Stochastic Volatility
    
    Key Features:
    • Monte Carlo simulation
    • Parameter estimation
    • Bond pricing
    • Yield curve modeling
    • Risk metrics
    • ML extensions
    
    Total simulation paths: {n_paths:,}
    Time horizon: {T} years
    Time step: {dt:.4f}
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.suptitle('CIR and Vasicek Interest Rate Models - Complete Analysis', y=0.98, fontsize=16)
    plt.show()
    
    print("\n\n7. PROJECT SUMMARY")
    print("-" * 30)
    
    print("✅ Successfully implemented comprehensive interest rate modeling framework!")
    print()
    print("Key Features Implemented:")
    print("• CIR and Vasicek models with analytical solutions")
    print("• Monte Carlo simulation with exact and Euler methods")
    print("• Parameter estimation using MLE and moment matching")
    print("• Bond pricing and yield curve calculation")
    print("• Regime-switching models using Hidden Markov Models")
    print("• Stochastic volatility modeling")
    print("• Real data integration via FRED and yfinance")
    print("• Comprehensive risk metrics and statistical analysis")
    print("• Model comparison and validation tools")
    print()
    print("Usage Examples:")
    print("• python examples/basic_simulation.py")
    print("• python examples/ml_comparison.py")
    print("• python main.py")
    print()
    print("The project is ready for quantitative analysis and research!")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nProject execution interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()
