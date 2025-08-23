"""Configuration for Interest Rate Models"""

import os
from typing import Optional

FRED_API_KEY: Optional[str] = None

# Try to import local configuration (if it exists)
try:
    from config_local import *
    print("Loaded configuration from config_local.py")
except ImportError:
    print("No config_local.py found. Using default configuration.")
    print("Copy config_local.py.template to config_local.py and add your API keys.")

FRED_TREASURY_SERIES = {
    'DGS1MO': '1M',
    'DGS3MO': '3M', 
    'DGS6MO': '6M',
    'DGS1': '1Y',
    'DGS2': '2Y',
    'DGS3': '3Y',
    'DGS5': '5Y',
    'DGS7': '7Y',
    'DGS10': '10Y',
    'DGS20': '20Y',
    'DGS30': '30Y'
}

DEFAULT_CIR_PARAMS = {
    'kappa': 0.1,    # Mean reversion speed
    'theta': 0.05,   # Long-term mean
    'sigma': 0.1,    # Volatility
    'r0': 0.03       # Initial rate
}

DEFAULT_VASICEK_PARAMS = {
    'kappa': 0.1,    # Mean reversion speed
    'theta': 0.05,   # Long-term mean
    'sigma': 0.02,   # Volatility
    'r0': 0.03       # Initial rate
}

DEFAULT_SIMULATION_PARAMS = {
    'T': 5,          # Time horizon (years)
    'dt': 1/252,     # Time step (daily)
    'n_paths': 1000  # Number of simulation paths
}

def get_api_key(service: str) -> Optional[str]:
    if service.lower() == 'fred':
        return FRED_API_KEY or os.getenv('FRED_API_KEY')
    else:
        return None

def is_api_available(service: str) -> bool:
    if service.lower() == 'fred':
        return get_api_key('fred') is not None
    elif service.lower() == 'yfinance':
        return True  # yfinance doesn't require API key
    else:
        return False

def get_data_source_config(source_type: str) -> dict:
    if source_type == 'treasury_yields':
        return {
            'series_ids': FRED_TREASURY_SERIES,
            'default_source': 'fred'
        }
    else:
        return {}
