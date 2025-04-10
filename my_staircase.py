import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random

class stairCase():
    def __init__(self,
                 init_level=0.5,
                 init_step=0.1,
                 method="3D1U", 
                 step_factor=0.5,
                 max_reversals=100,
                 max_trials=50,
                 max_level=0.6,
                 sigma_level=None,
                 ):
        """
        Parameters
        ----------
        init_level : float
            The starting difficulty level.
        init_step : float
            The starting step size.
        method : str
            The staircase method to use. Options: "3D1U", "2D1U", "3U1D", "2U1D", etc.
        step_factor : float
            How much the step size changes on a reversal.
        min_level : float
            The minimum difficulty level.
        max_reversals : int
            Stop the staircase after this many reversals.
        max_trials : int
            The maximum number of trials to run before stopping.
        """

        #self.sign=sign_of_stair
        self.init_level = init_level

        self.level = init_level
        self.init_step = init_step
        self.step = init_step
        self.method = method
        self.step_factor = step_factor
        #self.min_level = min_level
        self.maxLevelNegative = -1*abs(max_level)


        self.trial_num = 0
        self.correct_counter = 0
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

    def update_staircase(self, is_correct: bool) -> bool:
        """
        Update the staircase with the response to the current trial.
        
        Parameters
        ----------
        is_correct : bool
            True if the response was correct, False otherwise.
        
        Returns
        -------
        bool
            True if the staircase continues, False if it's finished.
        """
        # Number of reversals after which step size won't get any smaller
        n_stop_step_change = 2

        # Helper to determine how many consecutive correct answers trigger a "down" step
        if self.method in ["3D1U", "3U1D",'3D1Ub','3U1Db']:
            n_up = 3
        elif self.method in ["2D1U", "2U1D"]:
            n_up = 2
        elif self.method in ["4D1U", "4U1D"]:
            n_up = 4
        elif self.method in ["1D1U", "1U1D"]:
            n_up = 1
        else:
            n_up = 3  # Default fallback

        # Handle method for measuring lapses (ignores other logic)
        if self.method == "lapse_rate":
            if len(self.lapse_levels) > 0:
                self.level = self.lapse_levels.pop()
                return True
            else:
                self.stair_stopped = True
                return False

        correctIncrement=np.abs(self.level+self.step)<np.abs(self.level)
        correctDecrease=np.abs(self.level-self.step)<np.abs(self.level)

        incrementLevelCloseness=abs(self.level+self.step)
        decrementLevelCloseness=abs(self.level-self.step)

        #print('increment',increment)
        
        if incrementLevelCloseness<decrementLevelCloseness:
            dirStep=1
        else:
            dirStep=-1
        

        print
        print('dirStep',dirStep)

        # -----------------
        # Main staircase logic
        # -----------------
        # doing wronggg
        if not is_correct: # if response is not correct
            # Incorrect response => go "up"
            self.correct_counter = 0
            self.stair_dirs.append(1)  # +1 indicates direction up

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

            # Now update the level using the (potentially) new step size
            
            if self.level-dirStep*self.step > self.maxLevelNegative:
                if dirStep<0: # if the stair is going up, we need to check if we are at the max level and set t min level(which is the max level in this case
                    self.level =self.level-dirStep*self.step
 
                elif dirStep>=0: 
                    self.level =self.level-dirStep*self.step
            else:
                self.level = self.maxLevelNegative

                # #self.level = -1*dirStep*self.max_level
                # if np.sign(self.init_level)==1:
                #     self.level =self.level+dirStep*self.step
                # # if the stair is going down, we need to check if we are at the min level and set t max level(which is the min level in this case)
                # else:
                # print('to')
                # self.level = self.maxLevelNegative

        elif is_correct: # if correct
            # Correct response => potentially go "down"
            self.correct_counter += 1

            # Only move "down" after n_up consecutive correct answers
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

                # Now update the level using the (potentially) new step size
                self.level =self.level+dirStep*self.step

                # # conditional update based on min level treshold
                # if abs(self.level) - np.abs(self.step) > 0:
                #     self.level =self.level-self.step
                # else:
                #     self.level =self.sign_of_stair*self.min_level

        # Record last response
        self.last_response = is_correct

        # Check stopping conditions: 
        # 1) If we've hit the max number of trials
        if self.trial_num == (self.max_trials):
            print('stair final trial ',self.trial_num)
            self.stair_stopped = True
            print("End of staircase: max trials reached.")
            return False

        # Otherwise, continue
        return True

            
        
##########----------------EXANPLE USAGE ------------------#########
stair = stairCase(init_level=-0.75, 
                  method="1U1D", 
                  max_reversals=300,
                  max_trials=20,
                    step_factor=0.671,
                    init_step=0.2,
                    max_level=0.80,
                    )
levels = []
trialNum=0
plt.figure(figsize=(10, 6))
while not stair.stair_stopped:
    level = stair.next_trial()
    levels.append(level)
    is_correct = random.random() >0.59 #np.abs(level)
    stair.update_staircase(is_correct)
    print(f"step: {stair.step}, c {is_correct}, rev: {stair.reversals}, trial: {trialNum}, level: {level}")

    # if stair.reversals>0 and stair.stair_dirs[-1]!=stair.stair_dirs[-2]:
    #     plt.plot(trialNum,levels[-1], 'o',color='black')
    if is_correct:
        plt.scatter(x=trialNum,y=levels[-1], color='green', s=30)
    else:
        plt.scatter(x=trialNum,y=levels[-1],color='red', s=30)
    trialNum+=1

plt.plot(levels, '-',label='Staircase', color='blue')
plt.xlabel('Trial')
plt.ylabel('Level (Difficulty)')
plt.title('3-Down-1-Up Staircase Procedure')
plt.axhline(np.mean(levels[-100:]), color='red', linestyle='dashed', label='Convergence Level (Last 100 Trials)')
plt.axhline(0, color='black', linestyle='dashed', label='Zero Level')
print(f'convergence level: {levels[-1]}')
plt.legend()
plt.show()







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
#         is_correct = random.random() > (0.5 - 0.001*level)  # Higher level = easier = more correct
#         stair.update_staircase(is_correct)
        
#         # Plot each trial
#         if is_correct:
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