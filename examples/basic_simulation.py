"""Basic Simulation Example for CIR and Vasicek Models"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from models.cir_model import CIRModel
from models.vasicek_model import VasicekModel
from data.data_fetcher import InterestRateDataFetcher


def run_basic_simulation():
    print("Basic Interest Rate Model Simulation")
    print("=" * 50)
    
    # 1. Create models
    print("1. Creating CIR and Vasicek models...")
    
    cir_model = CIRModel(kappa=0.1, theta=0.05, sigma=0.1, r0=0.03)
    vasicek_model = VasicekModel(kappa=0.1, theta=0.05, sigma=0.02, r0=0.03)
    
    print(f"CIR Model: κ={0.1}, θ={0.05}, σ={0.1}")
    print(f"Vasicek Model: κ={0.1}, θ={0.05}, σ={0.02}")
    
    # 2. Simulate paths
    print("\n2. Simulating interest rate paths...")
    T, dt, n_paths = 5, 1/252, 1000
    
    cir_times, cir_rates = cir_model.simulate_path(T, dt, n_paths, method='exact')
    vasicek_times, vasicek_rates = vasicek_model.simulate_path(T, dt, n_paths, method='exact')
    
    print(f"Simulated {n_paths} paths for {T} years with {dt:.4f} time step")
    
    # 3. Plot results
    print("\n3. Plotting simulation results...")
    cir_model.plot_simulation(cir_times, cir_rates, "CIR Model Simulation")
    vasicek_model.plot_simulation(vasicek_times, vasicek_rates, "Vasicek Model Simulation")
    
    # 4. Compare models
    print("\n4. Comparing CIR and Vasicek models...")
    vasicek_model.compare_with_cir(cir_model, cir_times, n_paths=500)
    
    # 5. Calculate bond prices and yields
    print("\n5. Calculating bond prices and yields...")
    current_rate = 0.05
    maturities = np.array([0.5, 1, 2, 5, 10, 20, 30])
    
    cir_prices = [cir_model.bond_price(current_rate, T) for T in maturities]
    vasicek_prices = [vasicek_model.bond_price(current_rate, T) for T in maturities]
    cir_yields = cir_model.yield_curve(current_rate, maturities)
    vasicek_yields = vasicek_model.yield_curve(current_rate, maturities)
    
    print(f"Current rate: {current_rate:.1%}")
    print("\nBond Prices:")
    print(f"{'Maturity':<8} {'CIR':<10} {'Vasicek':<10}")
    print("-" * 30)
    for i, T in enumerate(maturities):
        print(f"{T:<8.1f} {cir_prices[i]:<10.4f} {vasicek_prices[i]:<10.4f}")
    
    print("\nYields:")
    print(f"{'Maturity':<8} {'CIR':<10} {'Vasicek':<10}")
    print("-" * 30)
    for i, T in enumerate(maturities):
        print(f"{T:<8.1f} {cir_yields[i]:<10.4f} {vasicek_yields[i]:<10.4f}")
    
    # 6. Parameter estimation
    print("\n6. Parameter estimation example...")
    sample_rates = cir_model.create_sample_data(n_obs=1000, dt=1/252)
    
    print("Estimating CIR parameters from sample data...")
    cir_est_kappa, cir_est_theta, cir_est_sigma = cir_model.estimate_parameters(
        sample_rates, 1/252, method='moment'
    )
    
    print(f"True parameters: κ={0.1}, θ={0.05}, σ={0.1}")
    print(f"Estimated parameters: κ={cir_est_kappa:.4f}, θ={cir_est_theta:.4f}, σ={cir_est_sigma:.4f}")
    
    # 7. Yield curve analysis
    print("\n7. Yield curve analysis...")
    rates_to_plot = [0.02, 0.05, 0.08]
    maturities_fine = np.linspace(0.1, 30, 100)
    
    plt.figure(figsize=(15, 10))
    
    # CIR yield curves
    plt.subplot(2, 2, 1)
    for rate in rates_to_plot:
        yields = cir_model.yield_curve(rate, maturities_fine)
        plt.plot(maturities_fine, yields, label=f'r={rate:.1%}')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.title('CIR Model Yield Curves')
    plt.legend()
    plt.grid(True)
    
    # Vasicek yield curves
    plt.subplot(2, 2, 2)
    for rate in rates_to_plot:
        yields = vasicek_model.yield_curve(rate, maturities_fine)
        plt.plot(maturities_fine, yields, label=f'r={rate:.1%}')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.title('Vasicek Model Yield Curves')
    plt.legend()
    plt.grid(True)
    
    # Compare yield curves
    plt.subplot(2, 2, 3)
    current_rate = 0.05
    cir_yields = cir_model.yield_curve(current_rate, maturities_fine)
    vasicek_yields = vasicek_model.yield_curve(current_rate, maturities_fine)
    
    plt.plot(maturities_fine, cir_yields, 'b-', label='CIR', linewidth=2)
    plt.plot(maturities_fine, vasicek_yields, 'r-', label='Vasicek', linewidth=2)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.title(f'Yield Curve Comparison (r={current_rate:.1%})')
    plt.legend()
    plt.grid(True)
    
    # Risk metrics
    plt.subplot(2, 2, 4)
    cir_final_rates = cir_rates[:, -1]
    vasicek_final_rates = vasicek_rates[:, -1]
    
    plt.hist(cir_final_rates, bins=30, alpha=0.7, label='CIR', density=True)
    plt.hist(vasicek_final_rates, bins=30, alpha=0.7, label='Vasicek', density=True)
    plt.xlabel('Interest Rate')
    plt.ylabel('Density')
    plt.title('Final Rate Distributions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 8. Risk metrics
    print("\n8. Risk metrics...")
    print("Final Rate Statistics:")
    print(f"{'Metric':<15} {'CIR':<15} {'Vasicek':<15}")
    print("-" * 45)
    print(f"{'Mean':<15} {np.mean(cir_final_rates):<15.4f} {np.mean(vasicek_final_rates):<15.4f}")
    print(f"{'Std Dev':<15} {np.std(cir_final_rates):<15.4f} {np.std(vasicek_final_rates):<15.4f}")
    print(f"{'Min':<15} {np.min(cir_final_rates):<15.4f} {np.min(vasicek_final_rates):<15.4f}")
    print(f"{'Max':<15} {np.max(cir_final_rates):<15.4f} {np.max(vasicek_final_rates):<15.4f}")
    print(f"{'5% VaR':<15} {np.percentile(cir_final_rates, 5):<15.4f} {np.percentile(vasicek_final_rates, 5):<15.4f}")
    print(f"{'95% VaR':<15} {np.percentile(cir_final_rates, 95):<15.4f} {np.percentile(vasicek_final_rates, 95):<15.4f}")
    
    print("\nBasic simulation completed successfully!")


def run_data_analysis():
    """Run data analysis with real/simulated data"""
    print("\n" + "="*50)
    print("Data Analysis Example")
    print("="*50)
    
    fetcher = InterestRateDataFetcher()
    
    print("Fetching sample interest rate data...")
    yield_df = fetcher.fetch_treasury_yields(start_date='2023-01-01')
    
    if not yield_df.empty:
        short_rate = yield_df['10Y'].dropna()
        rates_array = short_rate.values
        
        print(f"Fetched {len(rates_array)} days of 10-year Treasury yield data")
        print(f"Current yield: {short_rate.iloc[-1]:.4f}")
        
        # Fit models to real data
        print("\nFitting models to real data...")
        
        cir_model = CIRModel(0.1, 0.05, 0.1)
        cir_kappa, cir_theta, cir_sigma = cir_model.estimate_parameters(rates_array, 1/252)
        
        vasicek_model = VasicekModel(0.1, 0.05, 0.02)
        vasicek_kappa, vasicek_theta, vasicek_sigma = vasicek_model.estimate_parameters(rates_array, 1/252)
        
        print("Estimated Parameters:")
        print(f"CIR: κ={cir_kappa:.4f}, θ={cir_theta:.4f}, σ={cir_sigma:.4f}")
        print(f"Vasicek: κ={vasicek_kappa:.4f}, θ={vasicek_theta:.4f}, σ={vasicek_sigma:.4f}")
        
        # Plot fitted models
        plt.figure(figsize=(15, 10))
        
        # Historical data
        plt.subplot(2, 2, 1)
        plt.plot(short_rate.index, short_rate.values, 'b-', label='Historical', linewidth=1)
        plt.axhline(cir_theta, color='r', linestyle='--', label=f'CIR θ={cir_theta:.4f}')
        plt.axhline(vasicek_theta, color='g', linestyle='--', label=f'Vasicek θ={vasicek_theta:.4f}')
        plt.xlabel('Date')
        plt.ylabel('10-Year Treasury Yield')
        plt.title('Historical Interest Rates')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Simulate with fitted parameters
        cir_fitted = CIRModel(cir_kappa, cir_theta, cir_sigma)
        vasicek_fitted = VasicekModel(vasicek_kappa, vasicek_theta, vasicek_sigma)
        
        times, cir_sim = cir_fitted.simulate_path(1, 1/252, 100)
        times, vasicek_sim = vasicek_fitted.simulate_path(1, 1/252, 100)
        
        # Compare distributions
        plt.subplot(2, 2, 2)
        plt.hist(short_rate.values, bins=30, density=True, alpha=0.7, label='Historical', color='blue')
        plt.hist(cir_sim[:, -1], bins=30, density=True, alpha=0.7, label='CIR', color='red')
        plt.hist(vasicek_sim[:, -1], bins=30, density=True, alpha=0.7, label='Vasicek', color='green')
        plt.xlabel('Interest Rate')
        plt.ylabel('Density')
        plt.title('Rate Distribution Comparison')
        plt.legend()
        plt.grid(True)
        
        # Yield curve comparison
        plt.subplot(2, 2, 3)
        current_rate = short_rate.iloc[-1]
        maturities = np.linspace(0.1, 10, 50)
        
        cir_yields = cir_fitted.yield_curve(current_rate, maturities)
        vasicek_yields = vasicek_fitted.yield_curve(current_rate, maturities)
        
        plt.plot(maturities, cir_yields, 'r-', label='CIR', linewidth=2)
        plt.plot(maturities, vasicek_yields, 'g-', label='Vasicek', linewidth=2)
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield')
        plt.title(f'Fitted Model Yield Curves (r={current_rate:.4f})')
        plt.legend()
        plt.grid(True)
        
        # Volatility comparison
        plt.subplot(2, 2, 4)
        historical_vol = np.std(np.diff(short_rate.values))
        cir_vol = np.std(cir_sim[:, -1])
        vasicek_vol = np.std(vasicek_sim[:, -1])
        
        vols = [historical_vol, cir_vol, vasicek_vol]
        labels = ['Historical', 'CIR', 'Vasicek']
        plt.bar(labels, vols)
        plt.ylabel('Volatility')
        plt.title('Volatility Comparison')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\nData analysis completed!")


if __name__ == "__main__":
    # Run basic simulation
    run_basic_simulation()
    
    # Run data analysis
    run_data_analysis()
