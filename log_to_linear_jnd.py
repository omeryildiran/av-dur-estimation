import numpy as np
z_75 = 0.6744897501960817  # z-score for 75% in a standard normal distribution



def log_to_jnd_s(sigma_log, mu_log) :
    """JND at 75% in seconds."""
    return standard_dur_s * np.exp(mu_log) * (np.exp(z_75 * sigma_log) - 1)



if __main__ == "__main__":
    arguments = {
        "sigma_log": 0.1,  # Example log-space standard deviation
        "mu_log": 0.0      # Example log-space mean (log of the standard duration)
    }

    standard_dur_s = arguments.get("standard_dur_s", 0.5)  # Default to 0.5 seconds if not provided
    sigma_log = arguments["sigma_log"]
    mu_log = arguments["mu_log"]
    import sys
    import os

    print(log_to_jnd_s(sigma_log, mu_log) * 1000)  # Print JND in milliseconds

    

