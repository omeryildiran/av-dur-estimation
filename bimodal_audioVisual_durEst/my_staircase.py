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
                 min_level=0.05,
                 max_reversals=100,
                 max_trials=50,
                 max_level=0.6,
                 sigma_level=0.1,
                 sign_of_stair=1):
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
        self.level = sign_of_stair*init_level
        self.init_step = sign_of_stair*init_step
        self.step = sign_of_stair*init_step
        self.method = method
        self.step_factor = step_factor
        self.min_level = min_level
        self.max_level = max_level
        self.trial_num = 0
        self.correct_counter = 0
        self.last_response = None
        self.sigma_level = sigma_level

        self.max_reversals = max_reversals
        self.reversals = 0
        self.history = []
        self.reversal_points = []

        self.stair_stopped = False
        self.stair_dirs = []
        self.max_trials = max_trials
        self.is_reversal=False
        self.sign_of_stair=sign_of_stair

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

        # -----------------
        # Main staircase logic
        # -----------------
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
                    self.step = self.init_step * (self.step_factor ** self.reversals)
                else:
                    self.step = self.init_step * (self.step_factor ** n_stop_step_change)

            # Now update the level using the (potentially) new step size
            if abs(self.level + self.step) <= abs(self.max_level):
                self.level =self.level+self.step
            else:
                self.level = self.sign_of_stair*self.max_level

        else: # if correct
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
                self.level =self.level-self.step
                # if abs(self.level) - np.abs(self.step) > self.min_level:
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

            
        
# ##########----------------EXANPLE USAGE ------------------#########
# stair = stairCase(init_level=0.95, 
#                   method="2U1D", 
#                   min_level=0.15, 
#                   max_reversals=300,
#                   max_trials=50,
#                     step_factor=0.671,
#                     init_step=0.2,
#                     max_level=0.8,
#                     sign_of_stair=1)
# levels = []
# trialNum=0
# plt.figure(figsize=(10, 6))
# while not stair.stair_stopped:
#     level = stair.next_trial()
#     levels.append(level)
#     is_correct = random.random() >0.3 #np.abs(level)
#     stair.update_staircase(is_correct)
#     print(f"step: {stair.step}, c {is_correct}, rev: {stair.reversals}, trial: {trialNum}, level: {level}")

#     if stair.reversals>0 and stair.stair_dirs[-1]!=stair.stair_dirs[-2]:
#         plt.plot(trialNum,levels[-1], 'o',color='black')
#     if is_correct:
#         plt.scatter(x=trialNum,y=levels[-1], color='green', s=30)
#     else:
#         plt.scatter(x=trialNum,y=levels[-1],color='red', s=30)
#     trialNum+=1

# plt.plot(levels, '-',label='Staircase', color='blue')
# plt.xlabel('Trial')
# plt.ylabel('Level (Difficulty)')
# plt.title('3-Down-1-Up Staircase Procedure')
# plt.axhline(np.mean(levels[-100:]), color='red', linestyle='dashed', label='Convergence Level (Last 100 Trials)')
# print(f'convergence level: {levels[-1]}')
# plt.legend()
# plt.show()







