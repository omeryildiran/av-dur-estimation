import numpy as np
import matplotlib.pyplot as plt

class DualStaircase:
    def __init__(self, initial_delta_dur=0, step_size=0.02, min_delta=0.01, max_delta=0.5, max_reversals=12):
        """
        Initialize dual staircases (2-down-1-up and 2-up-1-down).
        :param initial_delta_dur: Starting relative duration (seconds) for both staircases.
        :param step_size: Initial step size for adjusting delta_dur.
        :param min_delta: Minimum possible value for delta_dur.
        :param max_delta: Maximum possible value for delta_dur.
        :param max_reversals: Maximum number of reversals before stopping the staircase.
        """
        self.delta_dur_a = initial_delta_dur  # 2-down-1-up (70%)
        self.delta_dur_b = initial_delta_dur  # 2-up-1-down (30%)
        
        self.step_size_a = step_size
        self.step_size_b = step_size
        
        self.min_delta = min_delta
        self.max_delta = max_delta
        
        self.reversals_a = 0
        self.reversals_b = 0
        
        self.max_reversals = max_reversals
        
        self.correct_counter_a = 0
        self.correct_counter_b = 0
        
        self.last_response_a = None
        self.last_response_b = None
        
        self.history_a = []  # Track trial history for 2-down-1-up
        self.history_b = []  # Track trial history for 2-up-1-down

    def next_trial(self, staircase):
        """Return the current delta_dur for the selected staircase (a or b)."""
        if staircase == 'a':
            return self.delta_dur_a
        elif staircase == 'b':
            return self.delta_dur_b
        else:
            raise ValueError("Invalid staircase. Use 'a' for 2-down-1-up or 'b' for 2-up-1-down.")

    def update(self, response, staircase):
        """
        Update the specified staircase based on participant's response.
        :param response: 1 for correct, 0 for incorrect.
        :param staircase: 'a' for 2-down-1-up or 'b' for 2-up-1-down.
        """
        if staircase == 'a':  # 2-down-1-up staircase
            return self._update_2down1up(response)
        elif staircase == 'b':  # 2-up-1-down staircase
            return self._update_2up1down(response)
        else:
            raise ValueError("Invalid staircase. Use 'a' or 'b'.")

    def _update_2down1up(self, response):
        """Update the 2-down-1-up staircase."""
        is_reversal = False
        if self.last_response_a is not None and response != self.last_response_a:
            is_reversal = True
            self.reversals_a += 1

        # Update logic
        if response == 1:  # Correct
            self.correct_counter_a += 1
            if self.correct_counter_a == 2:  # Two correct responses in a row
                self.delta_dur_a = max(self.min_delta, self.delta_dur_a - self.step_size_a)
                self.correct_counter_a = 0
        else:  # Incorrect
            self.delta_dur_a = min(self.max_delta, self.delta_dur_a + self.step_size_a)
            self.correct_counter_a = 0

        self.last_response_a = response
        self.history_a.append((len(self.history_a) + 1, self.delta_dur_a, response, is_reversal))

        # Stop criterion
        if self.reversals_a >= self.max_reversals:
            return False  # Stop this staircase
        return True

    def _update_2up1down(self, response):
        """Update the 2-up-1-down staircase."""
        is_reversal = False
        if self.last_response_b is not None and response != self.last_response_b:
            is_reversal = True
            self.reversals_b += 1

        # Update logic
        if response == 0:  # Incorrect
            self.correct_counter_b += 1
            if self.correct_counter_b == 2:  # Two incorrect responses in a row
                self.delta_dur_b = max(self.min_delta, self.delta_dur_b + self.step_size_b)
                self.correct_counter_b = 0
        else:  # Correct
            self.delta_dur_b = min(self.max_delta, self.delta_dur_b - self.step_size_b)
            self.correct_counter_b = 0

        self.last_response_b = response
        self.history_b.append((len(self.history_b) + 1, self.delta_dur_b, response, is_reversal))

        # Stop criterion
        if self.reversals_b >= self.max_reversals:
            return False  # Stop this staircase
        return True

    def plot_staircases(self):
        """Plot the progress of both staircases."""
        plt.figure(figsize=(12, 6))

        # Plot staircase a (2-down-1-up)
        trials_a, deltas_a, responses_a, reversals_a = zip(*self.history_a)
        plt.plot(trials_a, deltas_a, label='2-Down-1-Up (70%)', marker='o', color='blue')

        # Plot staircase b (2-up-1-down)
        trials_b, deltas_b, responses_b, reversals_b = zip(*self.history_b)
        plt.plot(trials_b, deltas_b, label='2-Up-1-Down (30%)', marker='o', color='red')

        plt.xlabel('Trial Number')
        plt.ylabel('Delta Duration (s)')
        plt.title('Dual Staircase Progression')
        plt.legend()
        plt.show()

# Initialize the dual staircase
staircase = DualStaircase(initial_delta_dur=0, step_size=0.02, max_reversals=12)

# Simulate trials
np.random.seed(42)
for trial in range(50):
    # Alternate between staircases
    staircase_type = 'a' if trial % 2 == 0 else 'b'
    current_delta = staircase.next_trial(staircase_type)
    
    # Simulate response: probability of choosing test decreases as delta decreases
    if staircase_type == 'a':  # 70% point staircase
        response = 1 if np.random.rand() < (0.7 + 0.3 * current_delta / 0.3) else 0
    else:  # 30% point staircase
        response = 1 if np.random.rand() < (0.3 + 0.3 * current_delta / 0.3) else 0

    continue_staircase = staircase.update(response, staircase_type)
    if not continue_staircase:
        print(f"Stopping staircase {staircase_type}. Reversals reached limit.")

# Plot the staircase progression
staircase.plot_staircases()
