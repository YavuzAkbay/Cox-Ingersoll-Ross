#!/usr/bin/env python3
"""
Quick Start Script for Cox-Ingersoll-Ross Interest Rate Models

This script provides an easy way to run the application with different options.
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Cox-Ingersoll-Ross Interest Rate Models')
    parser.add_argument('--mode', choices=['basic', 'full', 'optimized'], default='optimized',
                       help='Run mode: basic (fast test), full (complete analysis), optimized (balanced)')
    parser.add_argument('--examples', action='store_true',
                       help='Run example scripts instead of main application')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cox-Ingersoll-Ross Interest Rate Models")
    print("=" * 60)
    print()
    
    if args.examples:
        print("Running example scripts...")
        print()
        
        # Run basic simulation example
        print("1. Running basic simulation example...")
        os.system("python examples/basic_simulation.py")
        print()
        
        # Run ML comparison example if it exists
        if os.path.exists("examples/ml_comparison.py"):
            print("2. Running ML comparison example...")
            os.system("python examples/ml_comparison.py")
        else:
            print("2. ML comparison example not found, skipping...")
        
        # Run HJM demo if it exists
        if os.path.exists("examples/hjm_demo.py"):
            print("3. Running HJM demo...")
            os.system("python examples/hjm_demo.py")
        else:
            print("3. HJM demo not found, skipping...")
            
    else:
        if args.mode == 'basic':
            print("Running basic test (fast)...")
            # Import and run basic test
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from models import CIRModel, VasicekModel
            from data import InterestRateDataFetcher
            from utils import calculate_risk_metrics
            
            # Quick test
            cir = CIRModel(0.1, 0.05, 0.1, r0=0.03)
            vasicek = VasicekModel(0.1, 0.05, 0.02, r0=0.03)
            
            print("Testing CIR and Vasicek models...")
            times, cir_rates = cir.simulate_path(1, 1/252, 50)
            times, vasicek_rates = vasicek.simulate_path(1, 1/252, 50)
            
            print(f"CIR final rate: {cir_rates[0, -1]:.4f}")
            print(f"Vasicek final rate: {vasicek_rates[0, -1]:.4f}")
            
            # Test bond pricing
            price = cir.bond_price(0.03, 1)
            print(f"1-year bond price: {price:.4f}")
            
            print("✅ Basic test completed successfully!")
            
        elif args.mode == 'full':
            print("Running full analysis (may take several minutes)...")
            os.system("python main.py")
            
        else:  # optimized
            print("Running optimized analysis (recommended)...")
            os.system("python main_optimized.py")
    
    print()
    print("=" * 60)
    print("Application completed!")
    print("=" * 60)
    print()
    print("Available options:")
    print("  python run_app.py --mode basic      # Quick test")
    print("  python run_app.py --mode optimized  # Balanced analysis (default)")
    print("  python run_app.py --mode full       # Complete analysis")
    print("  python run_app.py --examples        # Run example scripts")
    print()
    print("For more information, see README.md")

if __name__ == "__main__":
    main()
