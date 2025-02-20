from mealpy.evolutionary_based import GA
import os
import numpy as np
from mealpy import PermutationVar, BinaryVar, FloatVar

parameters = [
    {"epoch": 1000, "pop_size": 100},
]

def read_sets(file_path, pop_size, num_city):
    """
    Reads a file containing ranked sets and returns them as a list of lists.

    Parameters:
    - file_path (str): Path to the file containing the ranked sets.
    - pop_size (int): Expected population size (number of sets).
    - num_city (int): Expected number of cities (elements per set).

    Returns:
    - list: A list of ranked sets, where each set is a list of integers.

    Raises:
    - ValueError: If the file content does not match the expected dimensions.
    """
    sets = []
    with open(file_path, 'r') as file:
        for line in file:
            set = list(map(int, line.strip().split()))
            sets.append(set)

    # Validate the dimensions of the ranked sets
    if len(sets) != pop_size:
        raise ValueError(f"Expected {pop_size} sets, but found {len(sets)} in the file.")

    for set in sets:
        if len(set) != num_city:
            raise ValueError(f"Each set must have {num_city} elements. Found a set with {len(set)} elements.")

    return np.array(sets)

# Number of repetitions
num_trials = 10

def save_history_file(model, file_dir, file_prefix):
    """
    :param model: The optimization model containing the history
    :param file_dir: The directory where the history file will be saved
    :param file_prefix: Prefix for the history filename
    """
    history_file = os.path.join(file_dir, f"{file_prefix}_history.txt")

    with open(history_file, "w") as f:

        # for epoch in enumerate(model.history.list_global_best):
        for epoch in range(0, 1000):
            # fitness = agent.target.fitness, Global Fitness: {fitness}
            # solution = agent.solution, Solution: {solution}

            # Fetch additional metrics
            diversity = model.history.list_diversity[epoch]
            exploration = model.history.list_exploration[epoch]
            exploitation = model.history.list_exploitation[epoch]
            local_fitness = model.history.list_current_best_fit[epoch]
            global_fitness = model.history.list_global_best_fit[epoch]
            runtime = model.history.list_epoch_time[epoch]

            f.write(
                f"Epoch: {epoch+1}, Local Fitness: {local_fitness}, Global Fitness: {global_fitness}, Diversity: {diversity}, \
Exploration: {exploration}, Exploitation: {exploitation}, Runtime: {runtime}\n"
            )

    print(f"History saved to {history_file}")

from opfunu.cec_based.cec2021 import F12021

f1 = F12021(10, f_bias=0)

p1 = {
    "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
    "obj_func": f1.evaluate,
    "minmax": "min",
    "name": "F1",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f1.lb, ub=f1.ub)

# Base directories
os.chdir("../quantum-project/ga") 
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F12021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F12021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p1, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F12021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F12021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p1, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F12021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F12021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p1, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F12021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F12021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p1, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F22021

f2 = F22021(10, f_bias=0)

p2 = {
    "bounds": FloatVar(lb=f2.lb, ub=f2.ub),
    "obj_func": f2.evaluate,
    "minmax": "min",
    "name": "F2",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f2.lb, ub=f2.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F22021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F22021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p2, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F22021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F22021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p2, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F22021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F22021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p2, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F22021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F22021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p2, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F32021

f3 = F32021(10, f_bias=0)

p3 = {
    "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
    "obj_func": f3.evaluate,
    "minmax": "min",
    "name": "F3",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f3.lb, ub=f3.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F32021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F32021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p3, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F32021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F32021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p3, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F32021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F32021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p3, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F32021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F32021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p3, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F42021

f4 = F42021(10, f_bias=0)

p4 = {
    "bounds": FloatVar(lb=f4.lb, ub=f4.ub),
    "obj_func": f4.evaluate,
    "minmax": "min",
    "name": "F4",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f4.lb, ub=f4.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F42021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F42021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p4, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F42021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F42021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p4, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F42021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F42021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p4, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F42021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F42021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p4, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F52021

f5 = F52021(10, f_bias=0)

p5 = {
    "bounds": FloatVar(lb=f5.lb, ub=f5.ub),
    "obj_func": f5.evaluate,
    "minmax": "min",
    "name": "F5",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f5.lb, ub=f5.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F52021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F52021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p5, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F52021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F52021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p5, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F52021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F52021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p5, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F52021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F52021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p5, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F62021

f6 = F62021(10, f_bias=0)

p6 = {
    "bounds": FloatVar(lb=f6.lb, ub=f6.ub),
    "obj_func": f6.evaluate,
    "minmax": "min",
    "name": "F6",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f6.lb, ub=f6.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F62021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F62021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p6, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F62021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F62021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p6, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F62021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F62021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p6, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F62021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F62021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p6, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F72021

f7 = F72021(10, f_bias=0)

p7 = {
    "bounds": FloatVar(lb=f7.lb, ub=f7.ub),
    "obj_func": f7.evaluate,
    "minmax": "min",
    "name": "F7",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f7.lb, ub=f7.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F72021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F72021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p7, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F72021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F72021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p7, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F72021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F72021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p7, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F72021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F72021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p7, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F82021

f8 = F82021(10, f_bias=0)

p8 = {
    "bounds": FloatVar(lb=f8.lb, ub=f8.ub),
    "obj_func": f8.evaluate,
    "minmax": "min",
    "name": "F8",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f8.lb, ub=f8.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F82021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F82021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p8, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F82021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F82021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p8, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F82021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F82021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p8, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F82021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F82021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p8, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F92021

f9 = F92021(10, f_bias=0)

p9 = {
    "bounds": FloatVar(lb=f9.lb, ub=f9.ub),
    "obj_func": f9.evaluate,
    "minmax": "min",
    "name": "F9",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f9.lb, ub=f9.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F92021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F92021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p9, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F92021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F92021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p9, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F92021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F92021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p9, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F92021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F92021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p9, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


from opfunu.cec_based.cec2021 import F102021

f10 = F102021(10, f_bias=0)

p10 = {
    "bounds": FloatVar(lb=f10.lb, ub=f10.ub),
    "obj_func": f10.evaluate,
    "minmax": "min",
    "name": "F10",
    "save_population": True
}

# Create the problem instance
bounds = FloatVar(lb=f10.lb, ub=f10.ub)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F102021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F102021", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(base_starting_solution_dir, trial_name, "Mt19937.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist
    
    mt19937 = read_sets(file_path_mt19937, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p10, starting_solutions=mt19937)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F102021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F102021", "QuasiRandom")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(base_starting_solution_dir, trial_name, "QuasiRandom.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist
    
    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p10, starting_solutions=QuasiRandom)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F102021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F102021", "IBM")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_IBM = os.path.join(base_starting_solution_dir, trial_name, "IBM.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_IBM):
        print(f"Skipping {file_path_IBM} (File not found)")
        continue  # Skip if the file does not exist
    
    IBM = read_sets(file_path_IBM, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p10, starting_solutions=IBM)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
        


# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "CEC2021", "F102021")
base_history_dir = os.path.join("..", "history", "ga", "CEC2021", "F102021", "BeamSplitter")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(base_starting_solution_dir, trial_name, "BeamSplitter.txt")
    
    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist
    
    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 10)

    # Define the history directory for this trial
    trial_history_dir = os.path.join(base_history_dir, trial_name)
    os.makedirs(trial_history_dir, exist_ok=True)  # Ensure the directory exists

    # Loop through the parameter sets
    for idx, params in enumerate(parameters, start=1):
        # Generate a sub-directory and filename prefix based on epoch and population size
        file_prefix = f"e={params['epoch']}p={params['pop_size']}"
        file_dir = os.path.join(trial_history_dir, file_prefix)
        os.makedirs(file_dir, exist_ok=True)  # Ensure the directory exists

        # Initialize and solve the model with specific parameters
        model = GA.BaseGA(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(p10, starting_solutions=BeamSplitter)
        
        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)
        
        # Print the best results for each run
        print(f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}")
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")
