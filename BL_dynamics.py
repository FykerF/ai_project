import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

#only sample that is based on the original derrivation of the BL formula. 

def compute_risk_neutral_pdfs(option_prices_t, strikes_t, option_prices_t_minus_1, strikes_t_minus_1,
                              time_to_maturity_t, time_to_maturity_t_minus_1, risk_free_rate=0.01,
                              smoothing_factor=0, spline_degree=3):
    """
    Computes the risk-neutral PDFs at two time points using the Breeden-Litzenberger model,
    calculates statistical moments using SciPy, and plots the PDFs.

    Parameters:
    - option_prices_t: array-like, option prices at time t.
    - strikes_t: array-like, strike prices at time t.
    - option_prices_t_minus_1: array-like, option prices at time t-1.
    - strikes_t_minus_1: array-like, strike prices at time t-1.
    - time_to_maturity_t: float, time to maturity at time t (in years).
    - time_to_maturity_t_minus_1: float, time to maturity at time t-1 (in years).
    - risk_free_rate: float, annualized risk-free interest rate.
    - smoothing_factor: float, smoothing factor for spline interpolation.
    - spline_degree: int, degree of the spline (1 <= k <= 5).

    Returns:
    - results_t: dict, statistical moments and PDF at time t.
    - results_t_minus_1: dict, statistical moments and PDF at time t-1.
    """

    # Function to compute risk-neutral PDF and statistical moments using SciPy
    def compute_pdf_and_moments(option_prices, strikes, time_to_maturity):
        # Convert inputs to numpy arrays
        strikes = np.array(strikes)
        option_prices = np.array(option_prices)

        # Sort the strike prices and corresponding option prices
        sorted_indices = np.argsort(strikes)
        strikes = strikes[sorted_indices]
        option_prices = option_prices[sorted_indices]

        # Create spline interpolation of option prices with respect to strike prices
        spline = UnivariateSpline(strikes, option_prices, s=smoothing_factor, k=spline_degree)

        # Compute the second derivative across all strike prices
        second_derivs = spline.derivative(n=2)(strikes)

        # Compute the risk-neutral PDF using the Breeden-Litzenberger formula
        f_rn = np.exp(risk_free_rate * time_to_maturity) * second_derivs

        # Ensure the PDF is non-negative
        f_rn = np.maximum(f_rn, 0)

        # Normalize the PDF so that the area under the curve equals 1
        area = np.trapz(f_rn, strikes)
        f_rn_normalized = f_rn / area

        # Create a spline of the normalized PDF for integration
        pdf_spline = UnivariateSpline(strikes, f_rn_normalized, s=0, k=3, ext=1)

        # Define the limits of integration
        a = strikes[0]
        b = strikes[-1]

        # Compute statistical moments using scipy.integrate.quad
        mean, _ = quad(lambda x: x * pdf_spline(x), a, b, limit=2000)
        variance, _ = quad(lambda x: (x - mean) ** 2 * pdf_spline(x), a, b, limit=2000)
        std_dev = np.sqrt(variance)
        skewness_numerator, _ = quad(lambda x: (x - mean) ** 3 * pdf_spline(x), a, b, limit=2000)
        skewness = skewness_numerator / std_dev ** 3
        kurtosis_numerator, _ = quad(lambda x: (x - mean) ** 4 * pdf_spline(x), a, b, limit=2000)
        kurtosis = kurtosis_numerator / std_dev ** 4

        results = {
            'mean': mean,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        return results

    # Compute for time t
    results_t = compute_pdf_and_moments(option_prices_t, strikes_t, time_to_maturity_t)

    # Compute for time t-1
    results_t_minus_1 = compute_pdf_and_moments(option_prices_t_minus_1, strikes_t_minus_1,
                                                time_to_maturity_t_minus_1)  # dfgdfgfdgdgdfgfdg

    # Plot PDFs from both time points for comparison
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(results_t['strike_prices'], results_t['pdf'], label='Time t')
    plt.plot(results_t_minus_1['strike_prices'], results_t_minus_1['pdf'], label='Time t-1', linestyle='--')
    plt.title('Comparison of Risk-Neutral PDFs Over Time')
    plt.xlabel('Strike Price')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()'''

    return results_t, results_t_minus_1