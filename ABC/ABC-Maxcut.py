from mealpy.swarm_based import ABC
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
    with open(file_path, "r") as file:
        for line in file:
            set = list(map(int, line.strip().split()))
            sets.append(set)

    # Validate the dimensions of the ranked sets
    if len(sets) != pop_size:
        raise ValueError(
            f"Expected {pop_size} sets, but found {len(sets)} in the file."
        )

    for set in sets:
        if len(set) != num_city:
            raise ValueError(
                f"Each set must have {num_city} elements. Found a set with {len(set)} elements."
            )

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
os.chdir("D:/quantum-project/abc") 
import sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from Maxcut import *

filePath2 = os.path.join("..", "maxcut_problem", "brock200_2.clq.txt")
edges = load_maxcut_file(filePath2)

data1, num_nodes1 = prepare_maxcut_data(edges)
# print(data1)

# Create the problem instance
bounds = [BinaryVar(n_vars=num_nodes1, name="binary")]
problem = MaxCutProblem(bounds=bounds, minmax="max", data=data1, save_population=True)

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_2"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Brock200_2", "MT19937"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(
        base_starting_solution_dir, trial_name, "Mt19937.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist

    mt19937 = read_sets(file_path_mt19937, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=mt19937)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_2"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Brock200_2", "QuasiRandom"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(
        base_starting_solution_dir, trial_name, "QuasiRandom.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist

    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=QuasiRandom)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_2"
)
base_history_dir = os.path.join("..", "history", "ABC", "maxcut", "Brock200_2", "IBM")

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

    IBM = read_sets(file_path_IBM, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=IBM)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_2"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Brock200_2", "BeamSplitter"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(
        base_starting_solution_dir, trial_name, "BeamSplitter.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist

    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=BeamSplitter)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

filePath2 = os.path.join("..", "maxcut_problem", "brock200_4.clq.txt")
edges = load_maxcut_file(filePath2)

data1, num_nodes1 = prepare_maxcut_data(edges)
# print(data1)

# Create the problem instance
bounds = [BinaryVar(n_vars=num_nodes1, name="binary")]
problem = MaxCutProblem(bounds=bounds, minmax="max", data=data1, save_population=True)

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_4"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Brock200_4", "MT19937"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(
        base_starting_solution_dir, trial_name, "Mt19937.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist

    mt19937 = read_sets(file_path_mt19937, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=mt19937)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_4"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Brock200_4", "QuasiRandom"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(
        base_starting_solution_dir, trial_name, "QuasiRandom.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist

    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=QuasiRandom)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_4"
)
base_history_dir = os.path.join("..", "history", "ABC", "maxcut", "Brock200_4", "IBM")

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

    IBM = read_sets(file_path_IBM, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=IBM)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Brock200_4"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Brock200_4", "BeamSplitter"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(
        base_starting_solution_dir, trial_name, "BeamSplitter.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist

    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=BeamSplitter)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

filePath2 = os.path.join("..", "maxcut_problem", "C125.9.clq.txt")
edges = load_maxcut_file(filePath2)

data1, num_nodes1 = prepare_maxcut_data(edges)
# print(data1)

# Create the problem instance
bounds = [BinaryVar(n_vars=num_nodes1, name="binary")]
problem = MaxCutProblem(bounds=bounds, minmax="max", data=data1, save_population=True)

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "Maxcut", "C125.9")
base_history_dir = os.path.join("..", "history", "ABC", "maxcut", "C125.9", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(
        base_starting_solution_dir, trial_name, "Mt19937.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist

    mt19937 = read_sets(file_path_mt19937, 100, 125)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=mt19937)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "Maxcut", "C125.9")
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "C125.9", "QuasiRandom"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(
        base_starting_solution_dir, trial_name, "QuasiRandom.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist

    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 125)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=QuasiRandom)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "Maxcut", "C125.9")
base_history_dir = os.path.join("..", "history", "ABC", "maxcut", "C125.9", "IBM")

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

    IBM = read_sets(file_path_IBM, 100, 125)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=IBM)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join("..", "starting_solution", "Maxcut", "C125.9")
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "C125.9", "BeamSplitter"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(
        base_starting_solution_dir, trial_name, "BeamSplitter.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist

    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 125)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=BeamSplitter)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

filePath2 = os.path.join("..", "maxcut_problem", "gen200_p0.9_44.b.clq.txt")
edges = load_maxcut_file(filePath2)

data1, num_nodes1 = prepare_maxcut_data(edges)
# print(data1)

# Create the problem instance
bounds = [BinaryVar(n_vars=num_nodes1, name="binary")]
problem = MaxCutProblem(bounds=bounds, minmax="max", data=data1, save_population=True)

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "gen200_p0.9_44.b"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "gen200_p0.9_44.b", "MT19937"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(
        base_starting_solution_dir, trial_name, "Mt19937.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist

    mt19937 = read_sets(file_path_mt19937, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=mt19937)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "gen200_p0.9_44.b"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "gen200_p0.9_44.b", "QuasiRandom"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(
        base_starting_solution_dir, trial_name, "QuasiRandom.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist

    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=QuasiRandom)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "gen200_p0.9_44.b"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "gen200_p0.9_44.b", "IBM"
)

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

    IBM = read_sets(file_path_IBM, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=IBM)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "gen200_p0.9_44.b"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "gen200_p0.9_44.b", "BeamSplitter"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(
        base_starting_solution_dir, trial_name, "BeamSplitter.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist

    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 200)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=BeamSplitter)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

filePath2 = os.path.join("..", "maxcut_problem", "keller4.clq.txt")
edges = load_maxcut_file(filePath2)

data1, num_nodes1 = prepare_maxcut_data(edges)
# print(data1)

# Create the problem instance
bounds = [BinaryVar(n_vars=num_nodes1, name="binary")]
problem = MaxCutProblem(bounds=bounds, minmax="max", data=data1, save_population=True)

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Keller4"
)
base_history_dir = os.path.join("..", "history", "ABC", "maxcut", "Keller4", "MT19937")

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_mt19937 = os.path.join(
        base_starting_solution_dir, trial_name, "Mt19937.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_mt19937):
        print(f"Skipping {file_path_mt19937} (File not found)")
        continue  # Skip if the file does not exist

    mt19937 = read_sets(file_path_mt19937, 100, 171)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=mt19937)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Keller4"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Keller4", "QuasiRandom"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_QuasiRandom = os.path.join(
        base_starting_solution_dir, trial_name, "QuasiRandom.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_QuasiRandom):
        print(f"Skipping {file_path_QuasiRandom} (File not found)")
        continue  # Skip if the file does not exist

    QuasiRandom = read_sets(file_path_QuasiRandom, 100, 171)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=QuasiRandom)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Keller4"
)
base_history_dir = os.path.join("..", "history", "ABC", "maxcut", "Keller4", "IBM")

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

    IBM = read_sets(file_path_IBM, 100, 171)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=IBM)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

# Base directories
base_starting_solution_dir = os.path.join(
    "..", "starting_solution", "Maxcut", "Keller4"
)
base_history_dir = os.path.join(
    "..", "history", "ABC", "maxcut", "Keller4", "BeamSplitter"
)

# Ensure the base history directory exists
os.makedirs(base_history_dir, exist_ok=True)

# Loop through each trial directory (trial_1 to trial_9)
for trial_num in range(1, num_trials + 1):
    # Construct the input file path for the current trial
    trial_name = f"trial_{trial_num}"
    file_path_BeamSplitter = os.path.join(
        base_starting_solution_dir, trial_name, "BeamSplitter.txt"
    )

    # Read the corresponding starting solution
    if not os.path.exists(file_path_BeamSplitter):
        print(f"Skipping {file_path_BeamSplitter} (File not found)")
        continue  # Skip if the file does not exist

    BeamSplitter = read_sets(file_path_BeamSplitter, 100, 171)

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
        model = ABC.OriginalABC(epoch=params["epoch"], pop_size=params["pop_size"])
        history = model.solve(problem, starting_solutions=BeamSplitter)

        # Save the optimization history to a text file
        save_history_file(model, file_dir, file_prefix)

        # Print the best results for each run
        print(
            f"Trial {trial_num} - Run {idx}: Epoch={params['epoch']}, Pop Size={params['pop_size']}"
        )
        print(f"Best fitness: {model.g_best.target.fitness}")
        print(f"Best solution: {model.problem.decode_solution(model.g_best.solution)}")

