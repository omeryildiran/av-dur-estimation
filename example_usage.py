"""
Example usage of PsychometricFitter class

This script demonstrates how to use the new class-based interface
to fit psychometric functions to duration estimation data.
"""

from psychometric_fitter import PsychometricFitter
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fit psychometric functions using class interface')
    parser.add_argument('--no-error-bars', action='store_true',
                       help='Plot without error bars across participants')
    parser.add_argument('--data', default='mt_auditoryDurEst_2025-06-16_12h17.46.950.csv',
                       help='Data file to use (default: mt_auditoryDurEst...)')
    parser.add_argument('--fix-mu', action='store_true',
                       help='Fix mu (bias) to 0')
    parser.add_argument('--n-start', type=int, default=1,
                       help='Number of random starting points for optimization')
    args = parser.parse_args()
    
    # Initialize the fitter with data
    print(f"Loading data from {args.data}...")
    fitter = PsychometricFitter(
        data_path=args.data,
        fix_mu=args.fix_mu
    )
    
    # Fit the model
    print("\nFitting psychometric model...")
    result = fitter.fit(n_start=args.n_start, verbose=True)
    
    # Plot the results
    show_error_bars = not args.no_error_bars
    if show_error_bars:
        print("\nPlotting psychometric functions with error bars across participants...")
    else:
        print("\nPlotting psychometric functions without error bars...")
    
    fitter.plot_fitted_psychometric(show_error_bars=show_error_bars)
    
    # Example: Get parameters for a specific condition
    print("\n=== Example: Accessing fitted parameters ===")
    params = fitter.get_condition_params()
    print(f"Lambda (lapse rate): {params['lambda']:.3f}")
    print(f"Mu (bias): {params['mu']:.3f}")
    print(f"Sigma (discrimination): {params['sigma']:.3f}")
    
    # Example: Make predictions
    print("\n=== Example: Making predictions ===")
    test_duration = 0.6  # seconds
    standard_duration = 0.5  # seconds
    p_longer = fitter.predict(test_duration, standard_duration)
    print(f"P(test perceived longer than standard): {p_longer:.3f}")
    
    # Example: Calculate PSE
    pse_stats = fitter.calculate_pse_stats(
        params['mu'], params['sigma'], 
        params['lambda'], standard_duration
    )
    print(f"\nPSE shift: {pse_stats['pse_shift_pure']*1000:+.1f} ms")


if __name__ == "__main__":
    main()
