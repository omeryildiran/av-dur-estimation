import numpy as np

class audioDurationGen:
    def __init__(self, trial_per_condition=10):
        self.trial_per_condition = trial_per_condition
        self.standard_durations= np.array([0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 1.8])  # seconds
        self.relative_durations = np.array([-200, -100, -50, 0, 50, 100, 200]) / 1000  # Convert to seconds
        self.weber_fraction = 0.15 # ref (Hartcher-O’Brien & Alais, 2011)

    def gen_duration_matrix(self):
        # Calculate test durations and real relative durations using vectorized operations
        test_durations = np.sign(self.relative_durations) * self.standard_durations[:, None] * self.weber_fraction + self.relative_durations + self.standard_durations[:, None]
        real_relative_durations = test_durations - self.standard_durations[:, None]

        # Round the matrices
        test_durations = np.round(test_durations, 3)
        real_relative_durations = np.round(real_relative_durations, 3)

        # Create the durations matrix
        num_conditions = len(self.standard_durations) * len(self.relative_durations)
        durations_matrix = np.zeros((num_conditions, 4))

        # Fill the durations matrix
        durations_matrix[:, 0] = np.repeat(self.standard_durations, len(self.relative_durations))
        durations_matrix[:, 1] = np.tile(self.relative_durations, len(self.standard_durations))
        durations_matrix[:, 2] = real_relative_durations.flatten()
        durations_matrix[:, 3] = test_durations.flatten()

        # Repeat the matrix to get 5 trials per condition and shuffle
        durations_matrix = np.tile(durations_matrix, (self.trial_per_condition, 1))
        np.random.shuffle(durations_matrix)

        return durations_matrix
    



