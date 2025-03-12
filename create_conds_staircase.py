import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class audioDurationGen:
    def __init__(self, trial_per_condition=10, rise_conds=[0.05, 0.25], standard_durations=[0.5, 0.7],intens=3):
        self.trial_per_condition = trial_per_condition
        self.standard_durations = np.array(standard_durations)  # seconds
        self.rise_conds = rise_conds
        self.intens = intens  # intensity of the sound

    def gen_duration_matrix(self):
        """
        matrix columns:
        0: Standard durations
        1: Rise conditions
        2: Order of test
        3: Pre duration
        4: Post duration
        5: ISI duration
        6: Trial number

        """
        # Create the durations matrix
        num_conditions = len(self.standard_durations)
        conditions_matrix = np.zeros((num_conditions, 2))

        # Fill the durations matrix
        conditions_matrix[:, 0] = self.standard_durations  # Standard durations 0

        conditions_matrix = np.tile(conditions_matrix, (len(self.rise_conds), 1))

        # Add rise conditions (signal to noise ratio SNR)
        rise_durations = np.repeat(self.rise_conds, len(conditions_matrix[:, 0]) // len(self.rise_conds))
        rise_durations = np.round(rise_durations, 6)
        conditions_matrix[:, 1] = rise_durations  # Rise conditions 1
        
        # Extend by the trial per condition to get the final matrix
        conditions_matrix = np.tile(conditions_matrix, (self.trial_per_condition, 1))

        # test 1st or test 2nd order column exactly half of the time
        orders=np.tile(np.array([1, 2]), len(conditions_matrix) // 2).astype(int)
        conditions_matrix = np.column_stack((conditions_matrix, orders)) # Order of test 2

        # Shuffle the matrix to get random order
        np.random.shuffle(conditions_matrix)

        # Add pre_duration and post_duration
        pre_duration = np.random.normal(0.25, 0.05, len(conditions_matrix))
        post_duration = np.random.normal(0.25, 0.05, len(conditions_matrix))
        isi_duration = np.random.normal(0.25, 0.05, len(conditions_matrix))

        # Round the durations
        pre_duration = np.round(pre_duration, 6)
        post_duration = np.round(post_duration, 6)
        isi_duration = np.round(isi_duration, 6)

        conditions_matrix = np.column_stack((conditions_matrix, pre_duration))  # Pre duration 3
        conditions_matrix = np.column_stack((conditions_matrix, post_duration))  # Post duration 4
        conditions_matrix = np.column_stack((conditions_matrix, isi_duration))  # ISI duration 5

        # Add the trial number
        trial_num = np.arange(1, len(conditions_matrix) + 1)
        conditions_matrix = np.column_stack((conditions_matrix, trial_num))  # Trial number 5

        return conditions_matrix




# # Example usage
# gen = audioDurationGen(trial_per_condition=25, rise_conds=[0.1, 0.20])
# conditions_matrix = gen.gen_duration_matrix()
# print(conditions_matrix.shape)


# print(conditions_matrix[:,0])
# print("riseeeee")
# print(conditions_matrix[:,1])
# #print(conditions_matrix)

# # example
# gen = audioDurationGen(trial_per_condition=40,rise_conds=[0.1,0.20])
# conditions_matrix = gen.gen_duration_matrix()

# print(conditions_matrix.shape)
# #print(conditions_matrix)
# print(conditions_matrix.shape)
# import matplotlib.pyplot as plt
# # Standard durations vs. relative durations
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)

# plt.plot(conditions_matrix[:, 0], conditions_matrix[:, 3], 'o')
# print('min test duration:', conditions_matrix[:, 3].min())
# print('max test duration:', conditions_matrix[:, 3].max())
# print("avg test duration:", np.mean(conditions_matrix[:, 3]))
# print('avg total dur', np.mean(conditions_matrix[:, 0]+conditions_matrix[:, 3]+conditions_matrix[:, 7]+conditions_matrix[:, 8]+conditions_matrix[:, 9]))
# print('approximate experiment duration', np.mean(conditions_matrix[:, 0]+conditions_matrix[:, 3]+conditions_matrix[:, 7]+conditions_matrix[:, 8]+conditions_matrix[:, 9])*len(conditions_matrix)/60)
# # # min max lines
# #plt.axhline(y=conditions_matrix[:, 3].min(), color='r', linestyle='--')
# #plt.axhline(y=conditions_matrix[:, 3].max(), color='r', linestyle='--')

# plt.xlabel('Standard durations (s)')
# plt.ylabel('Real relative durations (s)')
# plt.title('Standard durations vs. real relative durations')
# plt.show()
    

