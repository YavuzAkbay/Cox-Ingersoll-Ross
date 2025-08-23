"""
Data Fetcher for Interest Rate Models

Simple data fetching utilities for interest rate data from FRED and Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import Optional
import requests
import sys
import os

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import get_api_key, is_api_available, get_data_source_config
except ImportError:
    def get_api_key(service: str) -> Optional[str]:
        return None
    
    def is_api_available(service: str) -> bool:
        return service.lower() == 'yfinance'
    
    def get_data_source_config(source_type: str) -> dict:
        return {}


class InterestRateDataFetcher:
    """Fetcher for interest rate data from multiple sources"""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize data fetcher"""
        self.fred_api_key = fred_api_key or get_api_key('fred')
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        if not is_api_available('fred'):
            print("Warning: FRED API key not found. Using sample data.")
            print("Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    def fetch_fred_data(self, series_id: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from FRED"""
        if not self.fred_api_key:
            warnings.warn("No FRED API key provided. Using sample data.")
            return self._generate_sample_data(start_date, end_date)
        
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'sort_order': 'asc'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
            
        try:
            response = requests.get(self.fred_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            observations = data['observations']
            
            dates = []
            values = []
            
            for obs in observations:
                if obs['value'] != '.':
                    dates.append(pd.to_datetime(obs['date']))
                    values.append(float(obs['value']) / 100)  # Convert to decimal
            
            df = pd.DataFrame({'value': values}, index=dates)
            df.index.name = 'date'
            
            return df
            
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            return self._generate_sample_data(start_date, end_date)
    
    def fetch_treasury_yields(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch Treasury yield curve data"""
        config = get_data_source_config('treasury_yields')
        series_ids = config.get('series_ids', {
            'DGS1MO': '1M', 'DGS3MO': '3M', 'DGS6MO': '6M', 'DGS1': '1Y',
            'DGS2': '2Y', 'DGS3': '3Y', 'DGS5': '5Y', 'DGS7': '7Y',
            'DGS10': '10Y', 'DGS20': '20Y', 'DGS30': '30Y'
        })
        
        yield_data = {}
        
        for series_id, maturity in series_ids.items():
            try:
                df = self.fetch_fred_data(series_id, start_date, end_date)
                if not df.empty:
                    yield_data[maturity] = df['value']
            except Exception as e:
                print(f"Error fetching {maturity} yield: {e}")
        
        if yield_data:
            yield_df = pd.DataFrame(yield_data)
            yield_df.index.name = 'date'
            return yield_df
        else:
            return self._generate_sample_yield_curve(start_date, end_date)
    
    def fetch_yfinance_data(self, symbol: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                if 'BOND' in symbol or 'TLT' in symbol or 'IEF' in symbol:
                    df['yield'] = 1 / df['Close'] - 1
                    return df[['Close', 'yield']]
                else:
                    return df[['Close']]
            else:
                return self._generate_sample_data(start_date, end_date)
                
        except Exception as e:
            print(f"Error fetching yfinance data: {e}")
            return self._generate_sample_data(start_date, end_date)
    
    def _generate_sample_data(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Generate sample interest rate data"""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2024-01-01'
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        dates = pd.date_range(start, end, freq='D')
        n_days = len(dates)
        
        # Vasicek-like process
        kappa = 0.1
        theta = 0.05
        sigma = 0.02
        dt = 1/365
        
        rates = np.zeros(n_days)
        rates[0] = theta
        
        for i in range(1, n_days):
            rates[i] = rates[i-1] + kappa * (theta - rates[i-1]) * dt + \
                      sigma * np.random.normal(0, np.sqrt(dt))
        
        df = pd.DataFrame({'value': rates}, index=dates)
        df.index.name = 'date'
        
        return df
    
    def _generate_sample_yield_curve(self, start_date: Optional[str] = None, 
                                   end_date: Optional[str] = None) -> pd.DataFrame:
        """Generate sample yield curve data"""
        if start_date is None:
            start_date = '2020-01-01'
        if end_date is None:
            end_date = '2024-01-01'
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='D')
        
        maturities = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
        
        # Base yield curve (upward sloping)
        base_yields = {
            '1M': 0.02, '3M': 0.025, '6M': 0.03, '1Y': 0.035, '2Y': 0.04,
            '3Y': 0.042, '5Y': 0.045, '7Y': 0.047, '10Y': 0.05, '20Y': 0.052, '30Y': 0.053
        }
        
        yield_data = {}
        n_days = len(dates)
        
        for maturity in maturities:
            base_yield = base_yields[maturity]
            trend = np.linspace(0, 0.02, n_days)
            noise = np.random.normal(0, 0.005, n_days)
            
            yields = base_yield + trend + noise
            yields = np.maximum(yields, 0.001)
            
            yield_data[maturity] = yields
        
        df = pd.DataFrame(yield_data, index=dates)
        df.index.name = 'date'
        
        return df
    
    def calculate_yield_curve_metrics(self, yield_df: pd.DataFrame) -> dict:
        """Calculate yield curve metrics"""
        if yield_df.empty:
            return {}
        
        latest_yields = yield_df.iloc[-1]
        
        metrics = {
            'slope_2y10y': latest_yields.get('10Y', 0) - latest_yields.get('2Y', 0),
            'slope_3m10y': latest_yields.get('10Y', 0) - latest_yields.get('3M', 0),
            'curvature': (latest_yields.get('2Y', 0) + latest_yields.get('10Y', 0)) / 2 - latest_yields.get('5Y', 0),
            'level': np.mean(list(latest_yields.values())),
            'volatility': np.std(list(latest_yields.values())),
            'max_yield': np.max(list(latest_yields.values())),
            'min_yield': np.min(list(latest_yields.values())),
            'date': yield_df.index[-1]
        }
        
        return metrics
    
    def plot_yield_curve(self, yield_df: pd.DataFrame, date: Optional[str] = None):
        """Plot yield curve"""
        import matplotlib.pyplot as plt
        
        if date is None:
            latest_yields = yield_df.iloc[-1]
            date_str = yield_df.index[-1].strftime('%Y-%m-%d')
        else:
            date_dt = pd.to_datetime(date)
            if date_dt in yield_df.index:
                latest_yields = yield_df.loc[date_dt]
                date_str = date
            else:
                print(f"Date {date} not found in data")
                return
        
        maturities = list(latest_yields.index)
        yields = list(latest_yields.values)
        
        plt.figure(figsize=(12, 8))
        
        # Plot yield curve
        plt.subplot(2, 2, 1)
        plt.plot(maturities, yields, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Maturity')
        plt.ylabel('Yield (%)')
        plt.title(f'Yield Curve - {date_str}')
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot yield curve evolution
        plt.subplot(2, 2, 2)
        if len(yield_df) > 1:
            recent_data = yield_df.tail(30)
            for maturity in ['1Y', '5Y', '10Y', '30Y']:
                if maturity in recent_data.columns:
                    plt.plot(recent_data.index, recent_data[maturity], label=maturity)
            plt.xlabel('Date')
            plt.ylabel('Yield (%)')
            plt.title('Yield Curve Evolution (30 days)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
        
        # Plot yield spreads
        plt.subplot(2, 2, 3)
        spreads = {
            '2Y-10Y': latest_yields.get('10Y', 0) - latest_yields.get('2Y', 0),
            '3M-10Y': latest_yields.get('10Y', 0) - latest_yields.get('3M', 0),
            '5Y-30Y': latest_yields.get('30Y', 0) - latest_yields.get('5Y', 0)
        }
        plt.bar(spreads.keys(), spreads.values())
        plt.ylabel('Spread (%)')
        plt.title('Key Yield Spreads')
        plt.grid(True)
        
        # Plot yield distribution
        plt.subplot(2, 2, 4)
        plt.hist(yields, bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Yield (%)')
        plt.ylabel('Frequency')
        plt.title('Yield Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of the data fetcher"""
    print("Interest Rate Data Fetcher Example")
    print("=" * 50)
    
    fetcher = InterestRateDataFetcher()
    
    print("Fetching Treasury yield data...")
    yield_df = fetcher.fetch_treasury_yields(start_date='2023-01-01')
    print(f"Fetched {len(yield_df)} days of yield data")
    
    metrics = fetcher.calculate_yield_curve_metrics(yield_df)
    print("\nYield Curve Metrics:")
    for key, value in metrics.items():
        if key != 'date':
            print(f"{key}: {value:.4f}")
    
    fetcher.plot_yield_curve(yield_df)


if __name__ == "__main__":
    main()
