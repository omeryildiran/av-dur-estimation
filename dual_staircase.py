import numpy as np
import matplotlib.pyplot as plt


def two_up_one_down_staircase(p: float, n_trials: int = 1000):
    """
    Simulate a 2-up, 1-down staircase procedure.
    
    Parameters:
    - p: probability of a correct response (0 < p < 1)
    - n_trials: total number of trials to simulate
    
    Returns:
    - levels: list of the level at each trial
    """
    level = 0  # Starting difficulty level
    correct_streak = 0  # Counts consecutive correct responses
    levels = []
    
    for _ in range(n_trials):
        response = np.random.rand() < p  # Simulate response with probability p
        if response:  # Correct response
            correct_streak += 1
            if correct_streak == 2:  # Two consecutive correct responses
                level += 1  # Increase difficulty
                correct_streak = 0  # Reset streak
        else:  # Incorrect response
            level -= 1  # Decrease difficulty
            correct_streak = 0  # Reset streak
        
        levels.append(level)
    
    return levels


def plot_staircase(levels, p):
    """Plot the staircase procedure."""
    plt.figure(figsize=(10, 6))
    plt.plot(levels, label=f'Staircase (p={p})', color='blue')
    plt.xlabel('Trial')
    plt.ylabel('Level (Difficulty)')
    plt.title('2-Up, 1-Down Staircase Procedure')
    plt.axhline(np.mean(levels[-100:]), color='red', linestyle='dashed', label='Convergence Level (Last 100 Trials)')
    plt.legend()
    plt.show()


def simulate_and_plot(p: float, n_trials: int = 1000):
    """
    Run the staircase simulation and plot the result.
    
    Parameters:
    - p: probability of a correct response (0 < p < 1)
    - n_trials: total number of trials to simulate
    """
    levels = two_up_one_down_staircase(p, n_trials)
    plot_staircase(levels, p)
    
    print(f"Average Level (Last 100 Trials): {np.mean(levels[-100:])}")


# Parameters for the simulation
p = 0.7  # Probability of a correct response
n_trials = 2000  # Number of trials

# Run the simulation
simulate_and_plot(p, n_trials)  # Visualize the staircase
