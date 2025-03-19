import numpy as np

def create_conditions_matrix(totalTrialN=270,standards=[0.5], background_levels=[0.1, 0.5, 0.85], conflicts=[+0.05, 0, -0.05]):
    standardDur = standards
    background_levels = background_levels
    conflicts = conflicts
    
    # Generate all combinations (background level x conflict)
    conditions = []
    for bn in background_levels:
        for conf in conflicts:
            for standard in standardDur:
                """Columns: 
                0: Standard durations, 
                1: background conditions, 
                2: Conflict A-V """
                conditions.append([standard, bn, conf])
    
    #expand the conditions to the total number of trials
    if totalTrialN % len(conditions) != 0:
        raise ValueError(f"Total number of trials must be a multiple of the number of unique conditions which is {len(conditions)}.")

    conditions = conditions * int(np.ceil(totalTrialN / len(conditions)))
    
    return np.array(conditions)

# # Example usage:
# conditions_matrix = create_conditions_matrix(totalTrialN=270,standards=[0.5], 
#                                              background_levels=[0.1, 0.5, 0.85], conflicts=[+0.05, 0, -0.05])

# print(conditions_matrix)
# print(f"Number of trials: {len(conditions_matrix)}")