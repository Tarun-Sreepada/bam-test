import os
import subprocess
from itertools import product

# Define the range of parameters
queue_depths = [2, 4, 8]
num_queues = [1, 2, 4]
block_sizes = [1, 2, 4]
num_blocks = [1,2,4,8]
page_sizes = [4096]
io_types = [0]
io_methods = [1]
num_io = 100000

# Output directory for results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

exec_path = os.path.join(os.getcwd(), "../build/bin/block-test")

# Function to run the command and save results
def run_command_and_save_results(exec_path, params):
    queue_depth, num_queue, block_size, num_block, page_size, io_type, io_method = params
    output_file = os.path.join(
        output_dir, 
        f"qd{queue_depth}_nq{num_queue}_bs{block_size}_nb{num_block}_ps{page_size}_it{io_type}_im{io_method}.txt"
    )

    command = [
        "sudo", exec_path,
        f"--device", "0",
        f"--controller", "0",
        f"--queue-depth", str(queue_depth),
        f"--num-queues", str(num_queue),
        f"--block-size", str(block_size),
        f"--num-blocks", str(num_block),
        f"--page-size", str(page_size),
        f"--io-type", str(io_type),
        f"--io-method", str(io_method),
        f"--num-io", str(num_io)
    ]

    try:
        with open(output_file, "w") as f:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(result.stdout)
            f.write("\\n--- STDERR ---\\n")
            f.write(result.stderr)
        print(f"Saved results to {output_file}")
    except Exception as e:
        print(f"Failed to run command: {e}")
        
# Generate all combinations of parameters
parameter_combinations = product(queue_depths, num_queues, block_sizes, num_blocks, page_sizes, io_types, io_methods)

# Run the commands for each combination
for params in parameter_combinations:
    run_command_and_save_results(exec_path, params)

print("All commands executed. Results are saved in the 'results' directory.")
