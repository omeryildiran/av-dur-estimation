import numpy as np

class audioDurationGen:
    def __init__(self, trial_per_condition=10,rise_conds=[0.1,2.5]):
        self.trial_per_condition = trial_per_condition
        self.standard_durations= np.array([ 0.85, 1.0, 1.25, 1.5])  # seconds
        self.relative_durations = np.array([-200, -100, -50, 0, 50, 100, 200]) / 1000  # Convert to seconds
        self.weber_fraction = 0.15 # ref (Hartcher-O’Brien & Alais, 2011)
        self.rise_conds = rise_conds

    def gen_duration_matrix(self):
        # Calculate test durations and real relative durations using vectorized operations
        test_durations = np.sign(self.relative_durations) * self.standard_durations[:, None] * self.weber_fraction + self.relative_durations + self.standard_durations[:, None]
        real_relative_durations = test_durations - self.standard_durations[:, None]

        # Round the matrices
        test_durations = np.round(test_durations, 3)
        real_relative_durations = np.round(real_relative_durations, 3)

        # Create the durations matrix
        num_conditions = len(self.standard_durations) * len(self.relative_durations)
        conditions_matrix = np.zeros((num_conditions, 4))

        # Fill the durations matrix
        conditions_matrix[:, 0] = np.repeat(self.standard_durations, len(self.relative_durations)) # Standard durations 0
        conditions_matrix[:, 1] = np.tile(self.relative_durations, len(self.standard_durations)) # Relative durations 1
        conditions_matrix[:, 2] = real_relative_durations.flatten() # Real relative durations 2
        conditions_matrix[:, 3] = test_durations.flatten() # Test durations 3

        conditions_matrix = np.tile(conditions_matrix, (len(self.rise_conds), 1))
        
        # Add rise conditions (signal to noise ratio SNR)
        rise_durations = np.repeat(self.rise_conds, len(conditions_matrix[:, 1]) // len(self.rise_conds))
        conditions_matrix = np.column_stack((conditions_matrix, rise_durations)) # Rise conditions 4

        # test 1st or test 2nd order column exactly half of the time
        orders=np.tile(np.array([1, 2]), len(conditions_matrix) // 2).astype(int)
        conditions_matrix = np.column_stack((conditions_matrix, orders)) # Order of test 5

        # extend by thne trial per condition to get the final matrix
        conditions_matrix = np.tile(conditions_matrix, (self.trial_per_condition, 1)) 
        
        # assign random intensity to each trial
        intensity = np.random.uniform(2, 10, len(conditions_matrix))
        conditions_matrix = np.column_stack((conditions_matrix, intensity)) # Intensity 6

        # shuffle the matrix to get random order
        np.random.shuffle(conditions_matrix)

        # add the trial number
        trial_num = np.arange(1, len(conditions_matrix)+1)
        conditions_matrix = np.column_stack((conditions_matrix, trial_num))

        
        return conditions_matrix
    

# example
gen = audioDurationGen(10,[0.1,2.5])
conditions_matrix = gen.gen_duration_matrix()

# conditions_matrix = gen.gen_duration_matrix()
# print(conditions_matrix.shape)
# #print(conditions_matrix)
# print(conditions_matrix.shape)
# import matplotlib.pyplot as plt
# # Standard durations vs. relative durations
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)

# plt.plot(conditions_matrix[:, 0], conditions_matrix[:, 3], 'o')

# # min max lines
# plt.axhline(y=conditions_matrix[:, 3].min(), color='r', linestyle='--')
# plt.axhline(y=conditions_matrix[:, 3].max(), color='r', linestyle='--')

# plt.xlabel('Standard durations (s)')
# plt.ylabel('Real relative durations (s)')
# plt.title('Standard durations vs. real relative durations')
# plt.show()
    
