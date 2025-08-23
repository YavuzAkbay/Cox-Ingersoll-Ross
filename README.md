# Cox-Ingersoll-Ross and Vasicek Interest Rate Models

**Author:** Yavuz Akbay
**Email:** [akbay.yavuz@gmail.com]

A clean, comprehensive Python framework for quantitative analysis of interest rate dynamics using the Cox-Ingersoll-Ross (CIR) and Vasicek models.

## 🎯 Overview

This project provides a complete toolkit for interest rate modeling and analysis, featuring:

- **Core Models**: CIR and Vasicek stochastic differential equations with optimized implementations
- **Real-time Data**: Integration with FRED and Yahoo Finance
- **ML Extensions**: Regime-switching and stochastic volatility models
- **Comprehensive Analysis**: Bond pricing, yield curves, risk metrics
- **Production Ready**: Well-tested, documented, and extensible codebase with numerical stability improvements

## 🚀 Key Features

### 📊 Core Interest Rate Models
- **CIR Model**: Mean-reverting model with square-root diffusion process
- **Vasicek Model**: Mean-reverting model with constant volatility
- **Analytical Solutions**: Closed-form bond pricing and yield curve calculations
- **Monte Carlo Simulation**: High-performance path simulation with optimized algorithms
- **Vectorized Operations**: Fast bond pricing for multiple maturities

### 🔗 Data Integration
- **FRED API**: Real-time Treasury yields and economic indicators
- **Yahoo Finance**: Bond ETF data and alternative yield sources
- **Historical Analysis**: Comprehensive backtesting and parameter estimation

### 🤖 Machine Learning Extensions
- **Regime-Switching Models**: Hidden Markov Models for structural breaks
- **Stochastic Volatility**: Advanced volatility modeling techniques
- **Model Comparison**: Statistical validation and performance metrics

### 📈 Analysis & Visualization
- **Yield Curve Modeling**: Term structure analysis and forecasting
- **Risk Metrics**: Duration, convexity, and VaR calculations
- **Model Calibration**: Maximum likelihood and moment matching estimation

## 📁 Project Structure

```
Cox-Ingersoll-Ross/
├── 📂 models/                 # Core model implementations
│   ├── cir_model.py          # CIR model with analytical solutions
│   ├── vasicek_model.py      # Vasicek model implementation
│   └── ml_extensions.py      # ML-enhanced models
├── 📂 data/                  # Data handling and fetching
│   └── data_fetcher.py       # FRED and Yahoo Finance integration
├── 📂 examples/              # Usage examples and tutorials
│   ├── basic_simulation.py   # Basic model demonstration
│   └── ml_comparison.py      # Machine learning model comparison
├── 📂 utils/                 # Utility functions and helpers
│   └── helpers.py            # Common utilities
├── 📄 main.py               # Main demonstration script
├── 📄 config.py             # Configuration and API settings
├── 📄 config_local.py       # Local configuration (API keys)
├── 📄 config_local.py.template # Template for local configuration
├── 📄 requirements.txt      # Python dependencies
└── 📄 README.md             # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YavuzAkbay/Cox-Ingersoll-Ross.git
   cd Cox-Ingersoll-Ross
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** (optional, for live data)
   ```bash
   cp config_local.py.template config_local.py
   # Edit config_local.py and add your FRED API key
   ```

### Dependencies

The project uses the following key libraries:
- **Numerical Computing**: `numpy`, `scipy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`
- **Data Sources**: `fredapi`, `yfinance`
- **Machine Learning**: `scikit-learn`, `statsmodels`

## 🚀 Quick Start

### Basic Usage

```python
from models import CIRModel, VasicekModel
import numpy as np

# Create models with realistic parameters
cir_model = CIRModel(kappa=0.15, theta=0.055, sigma=0.08, r0=0.045)
vasicek_model = VasicekModel(kappa=0.15, theta=0.055, sigma=0.015, r0=0.045)

# Simulate interest rate paths
times, rates = cir_model.simulate_path(T=5, dt=1/252, n_paths=1000)

# Calculate bond prices and yields
maturities = np.array([1, 2, 5, 10, 30])
yields = cir_model.yield_curve(0.045, maturities)
bond_price = cir_model.bond_price(0.045, 10)

print(f"10-year bond price: {bond_price:.4f}")
print(f"Yield curve: {yields}")
```

### Run Complete Demonstration

```bash
python main.py
```

This will run a comprehensive demonstration including:
- Model simulation and comparison
- Bond pricing and yield curve analysis
- Real-time data fetching and analysis
- Machine learning model evaluation

### Example Scripts

```bash
# Basic simulation example
python examples/basic_simulation.py

# Machine learning model comparison
python examples/ml_comparison.py
```

## 📚 Mathematical Framework

### CIR Model
The Cox-Ingersoll-Ross model follows the stochastic differential equation:

```
dr(t) = κ(θ - r(t))dt + σ√r(t)dW(t)
```

**Key Properties:**
- Mean-reverting process with speed κ
- Long-term mean level θ
- Volatility σ√r(t) (proportional to rate level)
- Ensures positive interest rates when 2κθ ≥ σ² (Feller condition)

### Vasicek Model
The Vasicek model follows:

```
dr(t) = κ(θ - r(t))dt + σdW(t)
```

**Key Properties:**
- Mean-reverting process with constant volatility σ
- Can generate negative rates (unlike CIR)
- Simpler analytical solutions
- Widely used in practice despite limitations

### Bond Pricing
Both models provide closed-form bond pricing:

```
P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
```

Where A(t,T) and B(t,T) are model-specific functions.

## 📊 Data Sources

### FRED (Federal Reserve Economic Data)
- **Treasury Yields**: 1M to 30Y maturities
- **Federal Funds Rate**: Short-term rate
- **Economic Indicators**: Various financial data

### Yahoo Finance
- **Bond ETFs**: TLT, IEF, SHY, BOND, AGG
- **Alternative Data**: When FRED is unavailable

## 🔧 Configuration

### API Setup
1. Get a free FRED API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Copy `config_local.py.template` to `config_local.py`
3. Add your API key to `config_local.py`

### Environment Variables
```bash
export FRED_API_KEY="your_api_key_here"
```

## 📈 Usage Examples

### 1. Model Simulation and Comparison

```python
from models import CIRModel, VasicekModel
import matplotlib.pyplot as plt

# Create models
cir = CIRModel(kappa=0.1, theta=0.05, sigma=0.1, r0=0.03)
vasicek = VasicekModel(kappa=0.1, theta=0.05, sigma=0.02, r0=0.03)

# Simulate paths
times, cir_rates = cir.simulate_path(T=10, dt=1/252, n_paths=1000)
_, vasicek_rates = vasicek.simulate_path(T=10, dt=1/252, n_paths=1000)

# Compare models
vasicek.compare_with_cir(cir, times, n_paths=500)
```

### 2. Real-time Data Analysis

```python
from data import InterestRateDataFetcher

# Fetch current yield data
fetcher = InterestRateDataFetcher()
yields = fetcher.fetch_treasury_yields(start_date='2020-01-01')

# Analyze yield curve
print(f"Current 10Y yield: {yields['10Y'].iloc[-1]:.2%}")
```

### 3. Parameter Estimation

```python
# Generate sample data
sample_data = cir_model.create_sample_data(n_obs=1000, dt=1/252)

# Estimate parameters
kappa_est, theta_est, sigma_est = cir_model.estimate_parameters(
    sample_data, dt=1/252, method='moment'
)

print(f"Estimated parameters: κ={kappa_est:.4f}, θ={theta_est:.4f}, σ={sigma_est:.4f}")
```

### 4. Machine Learning Extensions

```python
from models import RegimeSwitchingModel

# Create regime-switching model
rs_model = RegimeSwitchingModel(n_regimes=2)
rs_model.fit(historical_data)

# Forecast with regime changes
forecast = rs_model.forecast(steps=252)
```

## 🧪 Testing and Validation

The project includes comprehensive testing:

```bash
# Run basic functionality test
python -c "
from models import CIRModel, VasicekModel
from data import InterestRateDataFetcher
print('✅ All modules imported successfully')
"
```

## 🚀 Performance Optimizations

The project includes several performance optimizations:

- **Vectorized Bond Pricing**: 10x+ speedup for multiple maturities
- **Optimized Monte Carlo**: Pre-computed constants for faster simulation
- **Enhanced Parameter Estimation**: Improved numerical stability and accuracy
- **Numerical Safeguards**: Robust handling of edge cases and extreme parameters

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Author**: [YavuzAkbay] - For developing this comprehensive interest rate modeling framework
- **FRED**: Federal Reserve Economic Data for providing excellent financial data APIs
- **Academic Community**: For the foundational work on interest rate modeling
- **Open Source Community**: For the excellent Python libraries that make this project possible

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join the conversation in GitHub Discussions

## 📈 Roadmap

- [ ] **Multi-factor Models**: Implementation of multi-factor CIR and Vasicek models
- [ ] **Calibration Tools**: Advanced calibration methods and optimization
- [ ] **Risk Management**: VaR, CVaR, and stress testing capabilities
- [ ] **Web Interface**: Interactive web dashboard for model analysis

---

**⭐ Star this repository if you find it useful!**

*Built with ❤️ by [YavuzAkbay] for the quantitative finance community*
