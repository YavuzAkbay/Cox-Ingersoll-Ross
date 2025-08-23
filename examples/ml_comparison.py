"""Machine Learning Extensions Comparison Example"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.cir_model import CIRModel
from models.vasicek_model import VasicekModel
from models.ml_extensions import (
    RegimeSwitchingModel, 
    StochasticVolatilityModel, 
    NeuralNetworkForecaster,
    compare_models
)
from data.data_fetcher import InterestRateDataFetcher


def run_regime_switching_example():
    print("Regime-Switching Model Example")
    print("=" * 50)
    
    # Create sample data with regime changes
    print("1. Generating sample data with regime changes...")
    
    # Generate data with two regimes
    np.random.seed(42)
    n_obs = 2000
    dt = 1/252
    
    # Regime 1: Low volatility, low rates
    regime1_model = VasicekModel(0.1, 0.03, 0.01, r0=0.03)
    times1, rates1 = regime1_model.simulate_path(n_obs//2 * dt, dt, n_paths=1)
    
    # Regime 2: High volatility, high rates
    regime2_model = VasicekModel(0.15, 0.07, 0.03, r0=0.07)
    times2, rates2 = regime2_model.simulate_path(n_obs//2 * dt, dt, n_paths=1)
    
    # Combine regimes
    rates = np.concatenate([rates1[0, :], rates2[0, :]])
    
    print(f"Generated {len(rates)} observations with regime changes")
    print(f"Regime 1: θ={0.03}, σ={0.01}")
    print(f"Regime 2: θ={0.07}, σ={0.03}")
    
    # Fit regime-switching model
    print("\n2. Fitting regime-switching model...")
    regime_model = RegimeSwitchingModel(n_regimes=2, model_type='vasicek')
    regime_model.fit(rates, dt)
    
    # Plot regime analysis
    print("\n3. Plotting regime analysis...")
    regime_model.plot_regimes(rates, regime_model.regime_probs)
    
    # Simulate regime-switching process
    print("\n4. Simulating regime-switching process...")
    times, sim_rates, sim_regimes = regime_model.simulate(2, dt, n_paths=100)
    
    # Plot simulation results
    plt.figure(figsize=(15, 10))
    
    # Interest rate paths
    plt.subplot(2, 2, 1)
    for i in range(min(10, sim_rates.shape[0])):
        plt.plot(times, sim_rates[i, :], alpha=0.7)
    plt.xlabel('Time (years)')
    plt.ylabel('Interest Rate')
    plt.title('Regime-Switching Interest Rate Paths')
    plt.grid(True)
    
    # Regime transitions
    plt.subplot(2, 2, 2)
    for i in range(min(5, sim_regimes.shape[0])):
        plt.plot(times, sim_regimes[i, :], alpha=0.7, label=f'Path {i+1}' if i == 0 else "")
    plt.xlabel('Time (years)')
    plt.ylabel('Regime')
    plt.title('Regime Transitions')
    plt.legend()
    plt.grid(True)
    
    # Rate distribution by regime
    plt.subplot(2, 2, 3)
    for regime in range(regime_model.n_regimes):
        regime_mask = sim_regimes.flatten() == regime
        regime_rates = sim_rates.flatten()[regime_mask]
        plt.hist(regime_rates, bins=30, alpha=0.7, label=f'Regime {regime}')
    plt.xlabel('Interest Rate')
    plt.ylabel('Density')
    plt.title('Rate Distribution by Regime')
    plt.legend()
    plt.grid(True)
    
    # Transition matrix
    plt.subplot(2, 2, 4)
    if regime_model.transition_matrix is not None:
        import seaborn as sns
        sns.heatmap(regime_model.transition_matrix, annot=True, cmap='Blues')
        plt.title('Regime Transition Matrix')
    
    plt.tight_layout()
    plt.show()
    
    print("Regime-switching example completed!")


def run_stochastic_volatility_example():
    """Demonstrate stochastic volatility model"""
    print("\n" + "="*50)
    print("Stochastic Volatility Model Example")
    print("=" * 50)
    
    # Create sample data
    print("1. Generating sample data...")
    np.random.seed(42)
    n_obs = 1000
    dt = 1/252
    
    # Generate data with time-varying volatility
    rates = np.zeros(n_obs)
    volatilities = np.zeros(n_obs)
    
    # Initial values
    rates[0] = 0.05
    volatilities[0] = 0.02
    
    # Parameters
    kappa_r, theta_r = 0.1, 0.05
    kappa_v, theta_v, sigma_v = 0.2, 0.02, 0.01
    rho = 0.3
    
    # Simulate stochastic volatility process
    for i in range(1, n_obs):
        # Generate correlated shocks
        z1 = np.random.normal(0, 1)
        z2 = np.random.normal(0, 1)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        # Update volatility
        volatilities[i] = volatilities[i-1] + \
            kappa_v * (theta_v - volatilities[i-1]) * dt + \
            sigma_v * np.sqrt(volatilities[i-1]) * z2 * np.sqrt(dt)
        volatilities[i] = np.maximum(volatilities[i], 0.001)
        
        # Update interest rate
        rates[i] = rates[i-1] + \
            kappa_r * (theta_r - rates[i-1]) * dt + \
            volatilities[i] * z1 * np.sqrt(dt)
    
    print(f"Generated {n_obs} observations with stochastic volatility")
    
    # Fit stochastic volatility model
    print("\n2. Fitting stochastic volatility model...")
    sv_model = StochasticVolatilityModel(base_model_type='vasicek')
    sv_model.fit(rates, dt)
    
    print(f"Fitted parameters:")
    print(f"Rate process: κ={sv_model.kappa_r:.4f}, θ={sv_model.theta_r:.4f}")
    print(f"Volatility process: κ={sv_model.kappa_v:.4f}, θ={sv_model.theta_v:.4f}, σ={sv_model.sigma_v:.4f}")
    print(f"Correlation: ρ={sv_model.rho:.4f}")
    
    # Simulate stochastic volatility process
    print("\n3. Simulating stochastic volatility process...")
    times, sim_rates, sim_vols = sv_model.simulate(1, dt, n_paths=100)
    
    # Plot simulation results
    sv_model.plot_simulation(times, sim_rates, sim_vols)
    
    # Compare with original data
    plt.figure(figsize=(15, 10))
    
    # Original vs simulated rates
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(rates)) * dt, rates, 'b-', label='Original', alpha=0.7)
    plt.plot(times, np.mean(sim_rates, axis=0), 'r-', label='Simulated Mean', linewidth=2)
    plt.fill_between(times, 
                    np.mean(sim_rates, axis=0) - np.std(sim_rates, axis=0),
                    np.mean(sim_rates, axis=0) + np.std(sim_rates, axis=0),
                    alpha=0.3, label='±1σ')
    plt.xlabel('Time (years)')
    plt.ylabel('Interest Rate')
    plt.title('Original vs Simulated Rates')
    plt.legend()
    plt.grid(True)
    
    # Original vs simulated volatilities
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(volatilities)) * dt, volatilities, 'b-', label='Original', alpha=0.7)
    plt.plot(times, np.mean(sim_vols, axis=0), 'r-', label='Simulated Mean', linewidth=2)
    plt.fill_between(times, 
                    np.mean(sim_vols, axis=0) - np.std(sim_vols, axis=0),
                    np.mean(sim_vols, axis=0) + np.std(sim_vols, axis=0),
                    alpha=0.3, label='±1σ')
    plt.xlabel('Time (years)')
    plt.ylabel('Volatility')
    plt.title('Original vs Simulated Volatilities')
    plt.legend()
    plt.grid(True)
    
    # Rate vs volatility scatter
    plt.subplot(2, 2, 3)
    plt.scatter(rates, volatilities, alpha=0.5, label='Original')
    plt.scatter(sim_rates.flatten(), sim_vols.flatten(), alpha=0.1, label='Simulated')
    plt.xlabel('Interest Rate')
    plt.ylabel('Volatility')
    plt.title('Rate vs Volatility')
    plt.legend()
    plt.grid(True)
    
    # Volatility clustering
    plt.subplot(2, 2, 4)
    rate_changes = np.diff(rates)
    plt.plot(np.arange(len(rate_changes)) * dt, np.abs(rate_changes), 'b-', alpha=0.7, label='Original')
    sim_changes = np.diff(sim_rates[0, :])
    plt.plot(times[1:], np.abs(sim_changes), 'r-', alpha=0.7, label='Simulated')
    plt.xlabel('Time (years)')
    plt.ylabel('|Rate Change|')
    plt.title('Volatility Clustering')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Stochastic volatility example completed!")


def run_neural_network_example():
    """Demonstrate neural network forecasting"""
    print("\n" + "="*50)
    print("Neural Network Forecasting Example")
    print("=" * 50)
    
    # Generate sample data
    print("1. Generating sample data...")
    np.random.seed(42)
    n_obs = 1000
    dt = 1/252
    
    # Generate interest rate data
    model = VasicekModel(0.1, 0.05, 0.02, r0=0.05)
    times, rates = model.simulate_path(n_obs * dt, dt, n_paths=1)
    rates = rates[0, :]
    
    print(f"Generated {len(rates)} observations")
    
    # Define maturities for yield curve
    maturities = np.array([0.5, 1, 2, 5, 10])
    
    # Fit neural network model
    print("\n2. Fitting neural network model...")
    nn_model = NeuralNetworkForecaster(sequence_length=20, hidden_size=32)
    
    try:
        history = nn_model.fit(rates, maturities, epochs=50)
        print("Neural network training completed!")
        
        # Make predictions
        print("\n3. Making predictions...")
        predictions = nn_model.predict(rates, maturities)
        
        if len(predictions) > 0:
            # Plot results
            plt.figure(figsize=(15, 10))
            
            # Training history
            if hasattr(history, 'history'):
                plt.subplot(2, 2, 1)
                plt.plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training History')
                plt.legend()
                plt.grid(True)
            
            # Yield curve predictions
            plt.subplot(2, 2, 2)
            # Get actual yield curves for comparison
            actual_yields = []
            for i in range(20, len(rates)):
                yields = model.yield_curve(rates[i], maturities)
                actual_yields.append(yields)
            actual_yields = np.array(actual_yields)
            
            if len(predictions) > 0 and len(actual_yields) > 0:
                min_len = min(len(predictions), len(actual_yields))
                for i in range(min(5, min_len)):
                    plt.plot(maturities, predictions[i], 'r-', alpha=0.7, label='Predicted' if i == 0 else "")
                    plt.plot(maturities, actual_yields[i], 'b-', alpha=0.7, label='Actual' if i == 0 else "")
                plt.xlabel('Maturity (years)')
                plt.ylabel('Yield')
                plt.title('Yield Curve Predictions')
                plt.legend()
                plt.grid(True)
            
            # Prediction accuracy
            plt.subplot(2, 2, 3)
            if len(predictions) > 0 and len(actual_yields) > 0:
                min_len = min(len(predictions), len(actual_yields))
                mse = np.mean((predictions[:min_len] - actual_yields[:min_len])**2, axis=1)
                plt.plot(mse)
                plt.xlabel('Time Step')
                plt.ylabel('Mean Squared Error')
                plt.title('Prediction Accuracy')
                plt.grid(True)
            
            # Interest rate evolution
            plt.subplot(2, 2, 4)
            plt.plot(times, rates, 'b-', label='Interest Rate')
            plt.xlabel('Time (years)')
            plt.ylabel('Rate')
            plt.title('Interest Rate Evolution')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print("Neural network example completed!")
        else:
            print("No predictions generated.")
            
    except Exception as e:
        print(f"Neural network training failed: {e}")
        print("This might be due to missing TensorFlow installation.")


def run_model_comparison():
    """Compare all models"""
    print("\n" + "="*50)
    print("Model Comparison Example")
    print("=" * 50)
    
    # Generate sample data
    print("1. Generating sample data...")
    np.random.seed(42)
    n_obs = 1000
    dt = 1/252
    
    # Generate data using CIR model
    base_model = CIRModel(0.1, 0.05, 0.1, r0=0.05)
    times, rates = base_model.simulate_path(n_obs * dt, dt, n_paths=1)
    rates = rates[0, :]
    
    print(f"Generated {len(rates)} observations using CIR model")
    
    # Create different models
    print("\n2. Creating models for comparison...")
    
    # Traditional models
    cir_model = CIRModel(0.1, 0.05, 0.1)
    vasicek_model = VasicekModel(0.1, 0.05, 0.02)
    
    # ML models
    regime_model = RegimeSwitchingModel(n_regimes=2, model_type='vasicek')
    sv_model = StochasticVolatilityModel(base_model_type='vasicek')
    
    # Dictionary of models
    models = {
        'CIR': cir_model,
        'Vasicek': vasicek_model,
        'Regime-Switching': regime_model,
        'Stochastic Volatility': sv_model
    }
    
    # Compare models
    print("\n3. Comparing models...")
    results = compare_models(models, rates, dt)
    
    # Print comparison results
    print("\nModel Comparison Results:")
    print(f"{'Model':<20} {'Mean':<10} {'Std':<10}")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:<20} {result['mean']:<10.4f} {result['std']:<10.4f}")
    
    print("\nModel comparison completed!")


def run_real_data_analysis():
    """Analyze real data with ML models"""
    print("\n" + "="*50)
    print("Real Data Analysis with ML Models")
    print("=" * 50)
    
    # Fetch real data
    print("1. Fetching real interest rate data...")
    fetcher = InterestRateDataFetcher()
    
    try:
        # Fetch Treasury yields
        yield_df = fetcher.fetch_treasury_yields(start_date='2020-01-01')
        
        if not yield_df.empty:
            # Use 10-year yield as proxy for short rate
            short_rate = yield_df['10Y'].dropna()
            rates = short_rate.values
            
            print(f"Fetched {len(rates)} days of 10-year Treasury yield data")
            print(f"Data range: {short_rate.index[0]} to {short_rate.index[-1]}")
            print(f"Current yield: {short_rate.iloc[-1]:.4f}")
            
            # Fit ML models
            print("\n2. Fitting ML models to real data...")
            
            # Regime-switching model
            regime_model = RegimeSwitchingModel(n_regimes=2, model_type='vasicek')
            regime_model.fit(rates, 1/252)
            
            # Stochastic volatility model
            sv_model = StochasticVolatilityModel(base_model_type='vasicek')
            sv_model.fit(rates, 1/252)
            
            # Plot results
            plt.figure(figsize=(15, 10))
            
            # Historical data with regime colors
            plt.subplot(2, 2, 1)
            if regime_model.regime_probs is not None:
                dominant_regime = np.argmax(regime_model.regime_probs, axis=1)
                colors = plt.cm.Set1(dominant_regime / regime_model.n_regimes)
                plt.scatter(range(len(rates)), rates, c=colors, alpha=0.6)
            else:
                plt.plot(rates)
            plt.xlabel('Time')
            plt.ylabel('10-Year Treasury Yield')
            plt.title('Historical Rates with Regime Detection')
            plt.grid(True)
            
            # Volatility analysis
            plt.subplot(2, 2, 2)
            rate_changes = np.diff(rates)
            rolling_vol = pd.Series(rate_changes).rolling(30).std()
            plt.plot(rolling_vol.index, rolling_vol.values, 'b-', label='Rolling Volatility')
            plt.axhline(sv_model.theta_v, color='r', linestyle='--', label=f'Long-term Vol: {sv_model.theta_v:.4f}')
            plt.xlabel('Time')
            plt.ylabel('Volatility')
            plt.title('Time-Varying Volatility')
            plt.legend()
            plt.grid(True)
            
            # Regime probabilities
            plt.subplot(2, 2, 3)
            if regime_model.regime_probs is not None:
                for regime in range(regime_model.n_regimes):
                    plt.plot(regime_model.regime_probs[:, regime], label=f'Regime {regime}')
                plt.xlabel('Time')
                plt.ylabel('Regime Probability')
                plt.title('Regime Probabilities')
                plt.legend()
                plt.grid(True)
            
            # Model comparison
            plt.subplot(2, 2, 4)
            models = {
                'Historical': None,
                'Regime-Switching': regime_model,
                'Stochastic Vol': sv_model
            }
            
            # Calculate statistics
            stats = []
            labels = []
            
            # Historical statistics
            stats.append([np.mean(rates), np.std(rates)])
            labels.append('Historical')
            
            # Model statistics
            for name, model in models.items():
                if model is not None and hasattr(model, 'simulate'):
                    try:
                        times, sim_rates, *_ = model.simulate(1, 1/252, 100)
                        stats.append([np.mean(sim_rates[:, -1]), np.std(sim_rates[:, -1])])
                        labels.append(name)
                    except:
                        pass
            
            if len(stats) > 1:
                means = [s[0] for s in stats]
                stds = [s[1] for s in stats]
                
                x = np.arange(len(labels))
                width = 0.35
                
                plt.bar(x - width/2, means, width, label='Mean', alpha=0.7)
                plt.bar(x + width/2, stds, width, label='Std Dev', alpha=0.7)
                plt.xlabel('Model')
                plt.ylabel('Value')
                plt.title('Model Statistics Comparison')
                plt.xticks(x, labels)
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print("\nReal data analysis completed!")
            
        else:
            print("No data fetched. Using sample data.")
            
    except Exception as e:
        print(f"Error in real data analysis: {e}")
        print("Using sample data instead.")


def main():
    """Run all ML examples"""
    print("Machine Learning Extensions for Interest Rate Models")
    print("=" * 60)
    
    # Run regime-switching example
    run_regime_switching_example()
    
    # Run stochastic volatility example
    run_stochastic_volatility_example()
    
    # Run neural network example
    run_neural_network_example()
    
    # Run model comparison
    run_model_comparison()
    
    # Run real data analysis
    run_real_data_analysis()
    
    print("\n" + "="*60)
    print("All ML examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
