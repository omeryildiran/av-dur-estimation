import random
import numpy as np
import matplotlib.pyplot as plt

class TwoDownOneUpStaircase:
    def __init__(self, initial_delta_dur=0.2, step_size=0.02, min_delta=0.01, max_delta=0.5, max_reversals=12):
        """
        Parameters:
        - initial_delta_dur: The starting relative duration (seconds).
        - step_size: How much delta_dur changes on a reversal (seconds).
        - min_delta: The minimum step size for delta_dur.
        - max_delta: The maximum possible value of delta_dur.
        - max_reversals: Stop the staircase after this many reversals.
        """
        self.delta_dur = initial_delta_dur
        self.step_size = step_size
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.max_reversals = max_reversals
        
        self.reversals = 0
        self.trial_num = 0
        self.correct_counter = 0
        self.last_response = None
        self.history = []  # Store (trial_num, delta_dur, response, is_reversal)
        self.reversal_points = []

    def next_trial(self):
        """Return the current delta_dur for the next trial."""
        return self.delta_dur

    def update(self, response):
        """
        Update the staircase based on the participant's response.
        - response = 1 if the participant got the trial correct.
        - response = 0 if the participant got the trial incorrect.
        """
        self.trial_num += 1

        # Check if this is a reversal
        is_reversal = False
        if self.last_response is not None and response != self.last_response:
            is_reversal = True
            self.reversals += 1
            self.reversal_points.append(self.delta_dur)

        # Update correct/incorrect logic for 2-down-1-up
        if response == 1:  # Correct
            self.correct_counter += 1
            if self.correct_counter == 2:  # Two correct in a row
                self.delta_dur = max(self.min_delta, self.delta_dur - self.step_size)
                self.correct_counter = 0
        else:  # Incorrect
            self.delta_dur = min(self.max_delta, self.delta_dur + self.step_size)
            self.correct_counter = 0

        # Record trial history
        self.history.append((self.trial_num, self.delta_dur, response, is_reversal))
        self.last_response = response

        # Stop if maximum number of reversals is reached
        if self.reversals >= self.max_reversals:
            print(f"Stopping staircase after {self.reversals} reversals.")
            return False  # Signal to stop the staircase
        return True  # Continue staircase

    def plot_staircase(self):
        """Plot the staircase to visualize how delta_dur changes over trials."""
        trial_nums, deltas, responses, reversals = zip(*self.history)
        plt.figure(figsize=(10, 5))
        plt.plot(trial_nums, deltas, label='Delta Dur (s)', marker='o', color='blue')
        for i, rev in enumerate(reversals):
            if rev:
                plt.scatter(trial_nums[i], deltas[i], color='red', s=50, label='Reversal' if i == 0 else None)
        plt.xlabel('Trial Number')
        plt.ylabel('Delta Duration (s)')
        plt.title('2-Down-1-Up Staircase')
        plt.legend()
        plt.show()


# Example usage
staircase = TwoDownOneUpStaircase(initial_delta_dur=0.3, step_size=0.02, max_reversals=12)

# Simulate participant responses (for illustration purposes)
np.random.seed(42)
for trial in range(100):
    current_delta = staircase.next_trial()
    
    # Simulate participant performance: more likely to be correct when delta is large
    response = 1 if random.random() < (0.5 + 0.5 * (current_delta / 0.3)) else 0
    print(f"Trial {trial+1}: Delta Duration = {current_delta:.3f}, Response = {response}")
    
    continue_staircase = staircase.update(response)
    if not continue_staircase:
        break

staircase.plot_staircase()
