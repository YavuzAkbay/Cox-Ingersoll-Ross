"""
Heath-Jarrow-Morton (HJM) Framework Demonstration

This script demonstrates the comprehensive HJM forward rate modeling framework,
including different volatility structures, forward rate surface simulation,
bond pricing, and advanced analysis tools.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    HJMForwardRateModel, 
    create_sample_hjm_model,
    constant_volatility,
    exponential_volatility,
    hump_volatility,
    linear_volatility,
    power_volatility
)


def demonstrate_volatility_structures():
    """Demonstrate different volatility structures"""
    print("=" * 60)
    print("HJM VOLATILITY STRUCTURES DEMONSTRATION")
    print("=" * 60)
    
    # Create models with different volatility structures
    volatility_types = ['constant', 'exponential', 'hump', 'linear', 'power']
    models = {}
    
    for vol_type in volatility_types:
        try:
            model = create_sample_hjm_model(n_factors=2, volatility_type=vol_type)
            models[vol_type] = model
            print(f"✅ Created {vol_type} volatility model")
        except Exception as e:
            print(f"❌ Error creating {vol_type} volatility model: {e}")
    
    # Plot volatility functions
    plt.figure(figsize=(15, 10))
    
    t = 0.0  # Current time
    T_range = np.linspace(0.25, 10, 100)
    
    for i, (vol_type, model) in enumerate(models.items()):
        plt.subplot(2, 3, i+1)
        
        # Plot volatility functions for each factor
        for factor in range(model.n_factors):
            volatilities = [model.volatility_function(t, T, factor) for T in T_range]
            plt.plot(T_range, volatilities, label=f'Factor {factor+1}', linewidth=2)
        
        plt.xlabel('Maturity (years)')
        plt.ylabel('Volatility')
        plt.title(f'{vol_type.capitalize()} Volatility')
        plt.legend()
        plt.grid(True)
    
    # Plot drift functions
    plt.subplot(2, 3, 6)
    for vol_type, model in models.items():
        drifts = [model.drift_function(t, T) for T in T_range]
        plt.plot(T_range, drifts, label=vol_type.capitalize(), linewidth=2)
    
    plt.xlabel('Maturity (years)')
    plt.ylabel('Drift')
    plt.title('HJM Drift Functions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle('HJM Volatility and Drift Functions', y=1.02, fontsize=16)
    plt.show()
    
    return models


def demonstrate_forward_rate_simulation():
    """Demonstrate forward rate surface simulation"""
    print("\n" + "=" * 60)
    print("FORWARD RATE SURFACE SIMULATION")
    print("=" * 60)
    
    # Create HJM model
    hjm_model = create_sample_hjm_model(n_factors=3, volatility_type='exponential')
    print(f"Created HJM model with {hjm_model.n_factors} factors")
    print(f"Maturity grid: {len(hjm_model.maturities)} points from {hjm_model.maturities[0]:.2f} to {hjm_model.maturities[-1]:.2f} years")
    
    # Simulate forward rates
    print("Simulating forward rate evolution...")
    times, forward_rates, spot_rates = hjm_model.simulate_forward_rates(
        T=5, dt=1/252, n_paths=200, method='euler'
    )
    
    print(f"Simulation completed:")
    print(f"  Time steps: {len(times)}")
    print(f"  Paths: {forward_rates.shape[0]}")
    print(f"  Maturities: {forward_rates.shape[2]}")
    
    # Analyze results
    analysis = hjm_model.analyze_model_properties(times, forward_rates, spot_rates)
    hjm_model.print_analysis(analysis)
    
    # Plot forward rate surface
    print("Generating forward rate surface plots...")
    hjm_model.plot_forward_rate_surface(times, forward_rates, path_idx=0)
    
    # Plot yield curve evolution
    print("Generating yield curve evolution plots...")
    hjm_model.plot_yield_curve_evolution(times, forward_rates, path_idx=0)
    
    return hjm_model, times, forward_rates, spot_rates


def demonstrate_bond_pricing():
    """Demonstrate bond pricing with HJM model"""
    print("\n" + "=" * 60)
    print("HJM BOND PRICING DEMONSTRATION")
    print("=" * 60)
    
    # Create model and simulate
    hjm_model = create_sample_hjm_model(n_factors=2, volatility_type='exponential')
    times, forward_rates, spot_rates = hjm_model.simulate_forward_rates(
        T=3, dt=1/252, n_paths=100, method='euler'
    )
    
    # Bond pricing at different times
    print("Bond Pricing Analysis:")
    print(f"{'Time':<8} {'1Y Bond':<10} {'2Y Bond':<10} {'5Y Bond':<10} {'10Y Bond':<10}")
    print("-" * 50)
    
    time_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    maturities = [1, 2, 5, 10]
    
    for time_idx in time_indices:
        t = times[time_idx]
        forward_curve = forward_rates[0, time_idx, :]
        
        bond_prices = []
        for maturity in maturities:
            price = hjm_model.bond_price(forward_curve, t, t + maturity)
            bond_prices.append(price)
        
        print(f"{t:<8.2f} {bond_prices[0]:<10.4f} {bond_prices[1]:<10.4f} {bond_prices[2]:<10.4f} {bond_prices[3]:<10.4f}")
    
    # Yield curve calculation
    print("\nYield Curve Analysis:")
    print(f"{'Maturity':<10} {'Initial Yield':<15} {'Final Yield':<15} {'Change':<10}")
    print("-" * 55)
    
    initial_yields = hjm_model.yield_curve(forward_rates[0, 0, :], 0)
    final_yields = hjm_model.yield_curve(forward_rates[0, -1, :], times[-1])
    
    for i, maturity in enumerate([0.5, 1, 2, 5, 10, 20]):
        if maturity <= hjm_model.maturities[-1]:
            idx = np.argmin(np.abs(hjm_model.maturities - maturity))
            initial_yield = initial_yields[idx]
            final_yield = final_yields[idx]
            change = final_yield - initial_yield
            
            print(f"{maturity:<10.1f} {initial_yield:<15.4f} {final_yield:<15.4f} {change:<10.4f}")
    
    return hjm_model, times, forward_rates


def demonstrate_multi_factor_analysis():
    """Demonstrate multi-factor analysis"""
    print("\n" + "=" * 60)
    print("MULTI-FACTOR HJM ANALYSIS")
    print("=" * 60)
    
    # Create models with different numbers of factors
    factor_counts = [1, 2, 3, 4]
    models = {}
    
    for n_factors in factor_counts:
        try:
            model = create_sample_hjm_model(n_factors=n_factors, volatility_type='exponential')
            models[n_factors] = model
            print(f"✅ Created {n_factors}-factor model")
        except Exception as e:
            print(f"❌ Error creating {n_factors}-factor model: {e}")
    
    # Compare simulations
    plt.figure(figsize=(15, 10))
    
    for i, (n_factors, model) in enumerate(models.items()):
        plt.subplot(2, 2, i+1)
        
        # Simulate with fewer paths for speed
        times, forward_rates, spot_rates = model.simulate_forward_rates(
            T=2, dt=1/252, n_paths=50, method='euler'
        )
        
        # Plot spot rate paths
        for path in range(min(10, spot_rates.shape[0])):
            plt.plot(times, spot_rates[path, :], alpha=0.3, color='blue')
        
        # Plot mean path
        mean_path = np.mean(spot_rates, axis=0)
        plt.plot(times, mean_path, 'red', linewidth=2, label='Mean')
        
        plt.xlabel('Time (years)')
        plt.ylabel('Spot Rate')
        plt.title(f'{n_factors}-Factor Model')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle('Multi-Factor HJM Model Comparison', y=1.02, fontsize=16)
    plt.show()
    
    # Analyze factor impact
    print("\nFactor Impact Analysis:")
    print(f"{'Factors':<8} {'Mean Rate':<12} {'Std Rate':<12} {'Min Rate':<12} {'Max Rate':<12}")
    print("-" * 60)
    
    for n_factors, model in models.items():
        times, forward_rates, spot_rates = model.simulate_forward_rates(
            T=1, dt=1/252, n_paths=100, method='euler'
        )
        
        final_rates = spot_rates[:, -1]
        print(f"{n_factors:<8} {np.mean(final_rates):<12.4f} {np.std(final_rates):<12.4f} {np.min(final_rates):<12.4f} {np.max(final_rates):<12.4f}")
    
    return models


def demonstrate_calibration():
    """Demonstrate model calibration"""
    print("\n" + "=" * 60)
    print("HJM MODEL CALIBRATION")
    print("=" * 60)
    
    # Create a model to generate "market" data
    true_model = create_sample_hjm_model(n_factors=2, volatility_type='exponential')
    times, forward_rates, spot_rates = true_model.simulate_forward_rates(
        T=1, dt=1/252, n_paths=1, method='euler'
    )
    
    # Generate market yield curve
    market_maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 20, 30])
    market_yields = []
    
    for maturity in market_maturities:
        if maturity <= times[-1]:
            # Find closest time point
            time_idx = np.argmin(np.abs(times - maturity))
            forward_curve = forward_rates[0, time_idx, :]
            yield_val = true_model.yield_curve(forward_curve, times[time_idx])
            # Find closest maturity in model
            maturity_idx = np.argmin(np.abs(true_model.maturities - maturity))
            market_yields.append(yield_val[maturity_idx])
        else:
            # Use final forward curve
            forward_curve = forward_rates[0, -1, :]
            yield_val = true_model.yield_curve(forward_curve, times[-1])
            maturity_idx = np.argmin(np.abs(true_model.maturities - maturity))
            market_yields.append(yield_val[maturity_idx])
    
    market_yields = np.array(market_yields)
    
    print("Market Yield Curve:")
    print(f"{'Maturity':<10} {'Yield':<10}")
    print("-" * 20)
    for maturity, yield_val in zip(market_maturities, market_yields):
        print(f"{maturity:<10.2f} {yield_val:<10.4f}")
    
    # Create model to calibrate
    test_model = create_sample_hjm_model(n_factors=2, volatility_type='exponential')
    
    # Calibrate model
    calibration_results = test_model.calibrate_to_market(
        market_yields, market_maturities, method='least_squares'
    )
    
    print(f"\nCalibration Results:")
    print(f"  Error: {calibration_results['error']:.6f}")
    print(f"  Calibrated: {calibration_results['calibrated']}")
    
    # Plot calibration results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(market_maturities, market_yields, 'bo-', label='Market', linewidth=2, markersize=8)
    plt.plot(market_maturities, calibration_results['model_yields'], 'ro-', label='Model', linewidth=2, markersize=8)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.title('Yield Curve Calibration')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    error = calibration_results['model_yields'] - market_yields
    plt.bar(market_maturities, error, alpha=0.7)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Calibration Error')
    plt.title('Calibration Errors')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(market_maturities, market_yields, 'bo-', label='Market', linewidth=2)
    plt.plot(market_maturities, calibration_results['model_yields'], 'ro-', label='Model', linewidth=2)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield')
    plt.title('Calibrated vs Market (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    relative_error = np.abs(error) / market_yields * 100
    plt.bar(market_maturities, relative_error, alpha=0.7, color='orange')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Calibration Errors')
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle('HJM Model Calibration Analysis', y=1.02, fontsize=16)
    plt.show()
    
    return test_model, calibration_results


def main():
    """Main demonstration function"""
    print("HEATH-JARROW-MORTON (HJM) FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. Demonstrate volatility structures
        models = demonstrate_volatility_structures()
        
        # 2. Demonstrate forward rate simulation
        hjm_model, times, forward_rates, spot_rates = demonstrate_forward_rate_simulation()
        
        # 3. Demonstrate bond pricing
        demonstrate_bond_pricing()
        
        # 4. Demonstrate multi-factor analysis
        multi_factor_models = demonstrate_multi_factor_analysis()
        
        # 5. Demonstrate calibration
        calibrated_model, calibration_results = demonstrate_calibration()
        
        print("\n" + "=" * 60)
        print("HJM DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("• Multiple volatility structures (constant, exponential, hump, linear, power)")
        print("• Multi-factor forward rate modeling")
        print("• Forward rate surface simulation and visualization")
        print("• Bond pricing and yield curve calculation")
        print("• Model calibration to market data")
        print("• Comprehensive statistical analysis")
        print()
        print("The HJM framework provides a sophisticated approach to interest rate modeling")
        print("with the ability to capture complex term structure dynamics.")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
