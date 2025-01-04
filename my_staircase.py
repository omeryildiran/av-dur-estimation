import random
import numpy as np
import matplotlib.pyplot as plt

class stairCase():
    def __init__(self,init_level=0.5,
                 init_step=0.1,
                 method="3D1U", 
                 step_factor=0.5,
                 min_level=0.001,
                 max_reversals=10,
                 max_trials=50):
        """Parameters
        init_level: The starting difficulty level.
        init_step: The starting step size.
        method: The staircase method to use. Options are "3D1U" (3-down, 1-up) and "2D1U" (2-down, 1-up).
        step_factor: How much the step size changes on a reversal.
        min_level: The minimum difficulty level.
        max_reversals: Stop the staircase after this many reversals."""
        self.level = init_level
        self.step = init_step
        self.method = method
        self.step_factor = step_factor
        self.min_level = min_level
        self.max_reversals = max_reversals
        self.reversals = 0
        self.trial_num = 0
        self.correct_counter = 0
        self.last_response = None
        self.history = []
        self.reversal_points = []
        self.stair_stopped=False
        self.init_step=init_step
        self.stair_dirs=[]
        self.max_trials=max_trials
        self.lapse_levels=[-0.7,-0.80, 0.8, 0.7]*5
        np.random.shuffle(self.lapse_levels)

    def next_trial(self):
        self.trial_num += 1
        return self.level

    def update_staircase(self,is_correct):
        """Update the staircase with the response to the current trial.
        This function takes a boolean argument is_correct, which is True if the response was correct and False if it was incorrect.
        The function returns True if the staircase is not finished and False if it is finished.
        Finish occurs when the number of reversals exceeds max_reversals."""

        
        if self.method=="3D1U" or self.method=="3U1D":
            n_up=3
        elif self.method=="2D1U" or self.method=="2U1D":
            n_up=2
        elif self.method=="4D1U" or self.method=="4U1D":
            n_up=4
        
        if self.method=="lapse_rate":
            if len(self.lapse_levels)>0:
                self.level=self.lapse_levels.pop()
                return True
            else:
                self.stair_stopped=True
                return False


        else:    
            n_stop_step_change=2
            # adjust the step size, step size is percentage respective to standard duration
            if self.reversals<n_stop_step_change:
                self.step = self.init_step *(self.step_factor**(self.reversals)) # Update the step size
            else:
                self.step = self.init_step * (self.step_factor**(n_stop_step_change))
            
            # if self.method=="3D1U":
            if not is_correct: # Incorrect response
                self.correct_counter=0 # Reset the correct counter

                #self.level= self.level+self.step if self.method == "3D1U" else self.level-self.step# Decrease difficulty by
                self.level += self.step if (np.abs(self.level+self.step))>self.min_level else self.min_level
                if len(self.stair_dirs)>=2 and self.stair_dirs[-1]!=self.stair_dirs[-2]: # Check if this is a reversal
                    self.reversals+=1 # Increment the reversal counter
                self.stair_dirs.append(1) # Append the direction to the list

            elif is_correct:
                self.correct_counter+=1 # Increment the correct counter

                if self.correct_counter==n_up: # Correct responses
                    self.correct_counter=0 # Reset the correct counter
                    #self.level= self.level-self.step #if self.method == "3D1U" else self.level+self.step # Increase difficulty by step
                    self.level = self.level-self.step if (np.abs(self.level-self.step))>self.min_level else self.min_level
                    if len(self.stair_dirs)>=2 and self.stair_dirs[-1]!=self.stair_dirs[-2]: # Check if this is a reversal
                        self.reversals+=1 # Increment the reversal counter
                    self.stair_dirs.append(-1) # Append the direction to the list

            
            # Check if the staircase is finished
            if self.trial_num<self.max_trials: 
                return True
            else:
                self.stair_stopped=True
                print("end of staircase")
                return False
            

# example
stair = stairCase(init_level=-0.001, init_step=0.2, 
                  method="3D1U", 
                  step_factor=0.5, 
                  min_level=0.001, 
                  max_reversals=3,
                  max_trials=50)
levels = []
trialNum=0
plt.figure(figsize=(10, 6))
while not stair.stair_stopped:
    trialNum+=1
    level = stair.next_trial()
    levels.append(level)
    is_correct = random.random() < np.abs(level)
    stair.update_staircase(is_correct)
    print(f"step: {stair.step}, c {is_correct}, rev: {stair.reversals}, trial: {trialNum}")
    if stair.reversals>1 and stair.stair_dirs[-1]!=stair.stair_dirs[-2]:
        plt.plot(levels, 'o-',color='red')
plt.plot(levels, 'o-',label='Staircase', color='blue')
plt.xlabel('Trial')
plt.ylabel('Level (Difficulty)')
plt.title('3-Down-1-Up Staircase Procedure')
plt.axhline(np.mean(levels[-100:]), color='red', linestyle='dashed', label='Convergence Level (Last 100 Trials)')
print(f'convergence level: {levels[-1]}')
plt.legend()
plt.show()







