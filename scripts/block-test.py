import os
import subprocess
from itertools import product

# Define the range of parameters
queue_depths = [1024]
num_queues = [1]
block_sizes = [1]
num_blocks = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
page_sizes = [4096]
io_types = [0]
io_methods = [1]
num_io = 100000

# Output directory for results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# get path of this script
script_path = os.path.abspath(__file__)
# go back 1 directory to get to the root of the project
script_path = os.path.dirname(script_path)

exec_path = os.path.join(script_path, "../build/bin/block-test")


def run_command_and_save_results(exec_path, params, output_dir, num_io):

    queue_depth, num_queue, block_size, num_block, page_size, io_type, io_method = params
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Base filename without the --clock flag
    base_filename = f"qd{queue_depth}_nq{num_queue}_bs{block_size}_nb{num_block}_ps{page_size}_it{io_type}_im{io_method}"
    
    # Define both command configurations
    command_configs = [
        {
            "use_clock": True,
            "filename_suffix": "clock",
            "args": [
                "sudo", exec_path + "_clock",
                "--device", "0",
                "--controller", "0",
                "--queue-depth", str(queue_depth),
                "--num-queues", str(num_queue),
                "--block-size", str(block_size),
                "--num-blocks", str(num_block),
                "--page-size", str(page_size),
                "--io-type", str(io_type),
                "--io-method", str(io_method),
                "--num-io", str(num_io),
            ]
        },
        # {
        #     "use_clock": False,
        #     "filename_suffix": "noclock",
        #     "args": [
        #         "sudo", exec_path + "_noclock",
        #         "--device", "0",
        #         "--controller", "0",
        #         "--queue-depth", str(queue_depth),
        #         "--num-queues", str(num_queue),
        #         "--block-size", str(block_size),
        #         "--num-blocks", str(num_block),
        #         "--page-size", str(page_size),
        #         "--io-type", str(io_type),
        #         "--io-method", str(io_method),
        #         "--num-io", str(num_io)
        #     ]
        # }
    ]
    
    for config in command_configs:
        # Construct the output filename based on whether --clock is used
        output_filename = f"{base_filename}_{config['filename_suffix']}.txt"
        output_file = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_file, "w") as f:
                # Execute the command
                result = subprocess.run(
                    config['args'], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True
                )
                
                # Write standard output to the file
                f.write(result.stdout)
                
                # If there's any standard error, write it under a separate header
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)
            
            print(f"Saved results to {output_file}")
        
        except Exception as e:
            print(f"Failed to run command for {'with' if config['use_clock'] else 'without'} --clock: {e}")


        
# Generate all combinations of parameters
parameter_combinations = product(queue_depths, num_queues, block_sizes, num_blocks, page_sizes, io_types, io_methods)

# Run the commands for each combination
for params in parameter_combinations:
    run_command_and_save_results(exec_path, params, output_dir, num_io)

print("All commands executed. Results are saved in the 'results' directory.")
