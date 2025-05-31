import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random

class stairCase():
    def __init__(self,
                 init_step=0.1,
                 method='3D1U',  # Options: '1U1D', '1D1U', '2U1D', '2D1U', 'lapse_rate'
                 step_factor=0.5,
                 max_reversals=100,
                 max_trials=50,
                 max_level=0.6,
                 ):
        
        # level at which the staircase starts
        if method[1]== 'U':
            self.level = abs(max_level)
        else:
            self.level = -1*abs(max_level)
        self.level=-0.9
        #self.sign=sign_of_stair
        self.method = method

        self.init_step = init_step
        self.step = init_step
        self.step_factor = step_factor
        #self.min_level = min_level
        self.maxLevelNegative = -1*abs(max_level)


        self.trial_num = 0
        self.correct_counter = 0
        self.notChooseTestCounter = 0
        self.last_response = None
        self.sigma_level = None

        self.max_reversals = max_reversals
        self.reversals = 0
        self.history = []
        self.reversal_points = []

        self.stair_stopped = False
        self.stair_dirs = []
        self.max_trials = max_trials
        self.is_reversal=False
        #self.sign_of_stair=sign_of_stair

        self.lapse_levels = [-0.55, 0.55,1] *300# big number so indeed we dont care about lapse rate here but we handle it in the experiment code itself.
        np.random.shuffle(self.lapse_levels)

    def next_trial(self):
        self.trial_num += 1
        return self.level

    def update_staircase(self, isChooseTest: bool) -> bool:
        """
        Update the staircase with the response to the current trial.
        
        Parameters
        ----------
        isChooseTest : bool
            True if the response was correct, False otherwise.
        
        Returns
        -------
        bool
            True if the staircase continues, False if it's finished.
        """
        # Number of reversals after which step size won't get any smaller
        n_stop_step_change = 2

        # Helper to determine how many consecutive correct answers trigger a "down" step


        # Handle method for measuring lapses (ignores other logic)
        if self.method == "lapse_rate":
            if len(self.lapse_levels) > 0:
                self.level = self.lapse_levels.pop()
                return True
            else:
                self.stair_stopped = True
                return False
        elif self.method != "lapse_rate":
            n_up=int(self.method[0])
            n_down=int(self.method[2])
            isUp = self.method[1] == 'U'  # True if the method is "Up"

            if isUp==True:
                dirStep=1
            else:
                dirStep=-1


        # -----------------
        # Main staircase logic
        # -----------------
        # doing wronggg
        if not isChooseTest: # if response is not correct

            # count down trials
            self.correct_counter = 0

            self.notChooseTestCounter +=1
            if self.notChooseTestCounter == n_down:
                self.notChooseTestCounter = 0
                self.stair_dirs.append(1)  # +1 indicates direction up

                # Incorrect response => 

                # Check if we have a reversal
                self.is_reversal = False
                if len(self.stair_dirs) >= 2 and self.stair_dirs[-1] != self.stair_dirs[-2]:
                    self.is_reversal = True
                    self.reversals += 1

                # If we have a reversal, update the step size immediately
                if self.is_reversal:
                    if self.reversals < n_stop_step_change:
                        self.step = self.init_step * (self.step_factor ** self.reversals) #pdate the step size
                    else:
                        self.step = self.init_step * (self.step_factor ** n_stop_step_change)
                
                # Update the level based on the direction
                if self.level-dirStep*self.step > self.maxLevelNegative:
                    # if we are at the max positive level
                    if self.level-dirStep*self.step >= 1:
                        self.level = 1
                    # if we are at the max level negative
                    else:
                        self.level =self.level-dirStep*self.step
                else:
                    self.level = self.maxLevelNegative


        elif isChooseTest: 
            self.correct_counter += 1
            self.notChooseTestCounter = 0

            # count up
            if self.correct_counter == n_up:
                self.correct_counter = 0
                self.stair_dirs.append(-1)  # -1 indicates direction down

                # Check if we have a reversal
                self.is_reversal = False
                if len(self.stair_dirs) >= 2 and self.stair_dirs[-1] != self.stair_dirs[-2]:
                    self.is_reversal = True
                    self.reversals += 1

                # If we have a reversal, update the step size immediately
                if self.is_reversal:
                    if self.reversals < n_stop_step_change:
                        self.step = self.init_step * (self.step_factor ** self.reversals)
                    else:
                        self.step = self.init_step * (self.step_factor ** n_stop_step_change)
                
                # Update the level based on the direction
                if self.level+dirStep*self.step >= 1: # if we are at the max positive level
                    self.level = 1  # Reset to the maximum level
                elif self.level+dirStep*self.step < self.maxLevelNegative: # if we are at the max level negative
                    self.level = self.maxLevelNegative  # Reset to the maximum level negative
                else:
                    self.level =self.level+dirStep*self.step


        # Record last response
        self.last_response = isChooseTest

        # Check stopping conditions: 
        # 1) If we've hit the max number of trials
        if self.trial_num == (self.max_trials):
            #print('stair final trial ',self.trial_num)
            self.stair_stopped = True
            #print("End of staircase: max trials reached.")
            return False

        # Otherwise, continue
        return True

method='2D1U'  # '1D2U', '2D1U'
      
        
##########----------------EXANPLE USAGE ------------------#########
stair = stairCase(max_reversals=300,
                  max_trials=50,
                    step_factor=0.671,
                    init_step=0.2,
                    max_level=0.90,
                    method=method  # '1D2U', '2D1U'
                    )
levels = []
trialNum=0
plt.figure(figsize=(10, 6))
while not stair.stair_stopped:
    level = stair.next_trial()
    levels.append(level)
    isChooseTest =  level - random.random() >0
    
    stair.update_staircase(isChooseTest)
    #print(f"step: {stair.step}, c {isChooseTest}, rev: {stair.reversals}, trial: {trialNum}, level: {level: .2f}")

    # if stair.reversals>0 and stair.stair_dirs[-1]!=stair.stair_dirs[-2]:
    #     plt.plot(trialNum,levels[-1], 'o',color='black')
    if isChooseTest:
        plt.scatter(x=trialNum,y=levels[-1], color='green', s=30)
    else:
        plt.scatter(x=trialNum,y=levels[-1],color='red', s=30)
    trialNum+=1
    

plt.plot(levels, '-',label='Staircase', color='blue')
plt.xlabel('Trial')
plt.ylabel('Level (Difficulty)')
#method='1D2U'  # '1D2U', '2D1U'
plt.title(f'{method} Staircase Procedure')
plt.axhline(np.mean(levels[-100:]), color='red', linestyle='dashed', label='Convergence Level (Last 100 Trials)')
plt.axhline(0, color='black', linestyle='dashed', label='Zero Level')
print(f'convergence level: {levels[-1]}')
plt.legend()
plt.ylim(-1, 1)
plt.show()


# Simulate and return average convergence level for a staircase method
def simulate_staircase(method, n_sim=100, max_trials=50, **kwargs):
    final_levels = []
    for _ in range(n_sim):
        stair = stairCase(method=method, max_trials=max_trials, **kwargs)
        while not stair.stair_stopped:
            level = stair.next_trial()
            # Simulate: higher level = easier = more likely correct
            isChooseTest =  level - random.random() >0.5
            stair.update_staircase(isChooseTest)
        final_levels.append(stair.level)
    return np.mean(final_levels)

# Example usage:
avg = simulate_staircase('1D2U', n_sim=100, step_factor=0.671, init_step=0.2, max_level=0.90)
avg2 = simulate_staircase('2D1U', n_sim=100, step_factor=0.671, init_step=0.2, max_level=0.90)

print(f"Average convergence level for 1D2U: {avg:.3f} | 2D1U: {avg2:.3f}")




# ##########----------------EXANPLE USAGE ------------------#########
# import matplotlib.pyplot as plt
# import numpy as np
# import random

# # Setup the figure with 2x2 subplots
# fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# axs = axs.flatten()

# # Define the methods to test
# methods = ["1U1D", "1D1U", "2U1D", "2D1U"]

# # Run each staircase method
# for i, method in enumerate(methods):
#     stair = stairCase(init_level=-0.95, 
#                      method=method, 
#                      max_reversals=300,
#                      max_trials=100,  # Using more trials to see patterns better
#                      step_factor=0.671,
#                      init_step=0.2,
#                      max_level=0.95)
    
#     levels = []
#     trialNum = 0
    
#     while not stair.stair_stopped:
#         if method 
#         level = stair.next_trial()
#         levels.append(level)
#         # Adjust probability based on level to make behavior more realistic
#         isChooseTest = random.random() > (0.5 - 0.001*level)  # Higher level = easier = more correct
#         stair.update_staircase(isChooseTest)
        
#         # Plot each trial
#         if isChooseTest:
#             axs[i].scatter(x=trialNum, y=levels[-1], color='green', s=30)
#         else:
#             axs[i].scatter(x=trialNum, y=levels[-1], color='red', s=30)
#         trialNum += 1
    
#     # Plot line connecting all levels
#     axs[i].plot(levels, '-', color='blue', alpha=0.7)
    
#     # Calculate mean of last several trials for convergence estimate
#     last_n = min(20, len(levels))
#     convergence = np.mean(levels[-last_n:])
    
#     # Add reference lines
#     axs[i].axhline(convergence, color='red', linestyle='dashed', 
#                   label=f'Convergence: {convergence:.2f}')
#     axs[i].axhline(0, color='black', linestyle='dashed')
    
#     # Set labels and title
#     axs[i].set_xlabel('Trial')
#     axs[i].set_ylabel('Level (Difficulty)')
#     axs[i].set_title(f'{method} Staircase Method')
#     axs[i].legend()
#     axs[i].set_ylim(-1, 1)

# plt.tight_layout()
# plt.savefig('staircase_methods_comparison.png')
# plt.show()

# print("Convergence levels:")
# for i, method in enumerate(methods):
#     print(f"{method}: {axs[i].get_lines()[1].get_ydata()[0]:.3f}")