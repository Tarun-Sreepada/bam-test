import os
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
from collections import defaultdict
from pathlib import Path


KILOBYTE = 1024
MEGABYTE = 1024 * KILOBYTE
GIGABYTE = 1024 * MEGABYTE
TERABYTE = 1024 * GIGABYTE

file_location = os.path.abspath(__file__)
file_dir = os.path.dirname(file_location)

bam_exec_path = os.path.join(file_dir, "../external/bam/build/bin/nvm-block-bench")
our_exec_path = os.path.join(file_dir, "../build/bin/block-test")

io_types = {
    0: "read",
    "read": 0,
    1: "write",
    "write": 1
}

io_methods = {
    1: "random",
    "random": 1,
    0: "sequential",
    "sequential": 0
}


def bytes_to_str(num_bytes):
    if num_bytes >= TERABYTE:
        return f"{num_bytes / TERABYTE:.2f} TB"
    elif num_bytes >= GIGABYTE:
        return f"{num_bytes / GIGABYTE:.2f} GB"
    elif num_bytes >= MEGABYTE:
        return f"{num_bytes / MEGABYTE:.2f} MB"
    elif num_bytes >= KILOBYTE:
        return f"{num_bytes / KILOBYTE:.2f} KB"
    else:
        return f"{num_bytes} B"
        
def bam_benchmark(exec_path, params, output_dir):
    output_file = os.path.join(output_dir, f"bam-qd{params['queue_depth']}-"
                                           f"nq{params['num_queue']}-"
                                           f"bs{params['block_size']}-"
                                           f"nb{params['num_blocks']}-"
                                           f"ps{params['page_size']}-"
                                           f"{params['io_type']}-"
                                           f"{params['io_method']}-"
                                           f"nio{params['num_io']}.txt")
    
    if os.path.exists(output_file):
        # if there is no clock file, then we need to rerun the benchmark
        data_flag = False
        with open(output_file, "r") as f:
            for line in f:
                if "Bandwidth:" in line or "Throughput:" in line:
                    data_flag = True
                    break
        if not data_flag:
            os.remove(output_file)
        else:
            print(f"Output file exists, skipping")
            return

    try:
        with open(output_file, "w") as f:
            params['io_type_id'] = io_types[params['io_type']]
            params['io_method_id'] = io_methods[params['io_method']]
            print(
                f"Running: {exec_path}\n"
                f"  --page_size   {params['page_size']}\n"
                f"  --blk_size    {params['block_size']}\n"
                f"  --queue_depth {params['queue_depth']}\n"
                f"  --gpu         {params['device']}\n"
                f"  --reqs        {params['num_io']}\n"
                f"  --access_type {params['io_type_id']}\n"
                f"  --random      {params['io_method_id']}\n"
                f"  --threads     {params['block_size'] * params['num_blocks']}"
            )

            result = subprocess.run(
                [
                    "sudo", exec_path,
                    "--page_size", str(params["page_size"]),
                    "--blk_size", str(params["block_size"]),
                    "--queue_depth", str(params["queue_depth"]),
                    "--gpu", str(params["device"]),
                    "--reqs", str(params["num_io"]),
                    "--access_type", str(params["io_type_id"]),
                    "--random", str(params["io_method_id"]),
                    "--threads", str(params["block_size"] * params["num_blocks"]),
                    "--reqs", str(params["num_io"])
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # timeout=180 # 3 minutes
            )
                
            f.write(result.stdout)
            
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)       
                
    except Exception as e:
        print(f"Failed to run command: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)               
    

    
def our_benchmark(exec_path, params, output_dir, clock=False):
    
    output_file = os.path.join(output_dir, f"our-qd{params['queue_depth']}-"
                                           f"nq{params['num_queue']}-"
                                           f"bs{params['block_size']}-"
                                           f"nb{params['num_blocks']}-"
                                           f"ps{params['page_size']}-"
                                           f"{params['io_type']}-"
                                           f"{params['io_method']}-"
                                           f"nio{params['num_io']}.txt")
    if clock:
        output_file = output_file.replace(".txt", "_clock.txt")
        exec_path = exec_path + "_clock"
        
    
    if os.path.exists(output_file):
        # if there is no clock file, then we need to rerun the benchmark
        data_flag = False
        with open(output_file, "r") as f:
            for line in f:
                if "Bandwidth:" in line or "Throughput:" in line:
                    data_flag = True
                    break
        if not data_flag:
            os.remove(output_file)
        else:
            print(f"Output file exists, skipping")
            return
        
    ssd_size = "3500 GB"
    
    try:
        with open(output_file, "w") as f:
            params['io_type_id'] = io_types[params['io_type']]
            params['io_method_id'] = io_methods[params['io_method']]
            
            print(
                f"Running command: {exec_path}\n"
                f"  --device        {params['device']}\n"
                f"  --controller    {params['controller']}\n"
                f"  --queue-depth   {params['queue_depth']}\n"
                f"  --num-queues    {params['num_queue']}\n"
                f"  --block-size    {params['block_size']}\n"
                f"  --num-blocks    {params['num_blocks']}\n"
                f"  --page-size     {params['page_size']}\n"
                f"  --io-type       {params['io_type_id']}\n"
                f"  --io-method     {params['io_method_id']}\n"
                f"  --num-io        {params['num_io']}\n"
                f"  --ssd-size      {ssd_size}"
            )

            result = subprocess.run(
                [
                    "sudo", exec_path,
                    "--device", str(params["device"]),
                    "--controller", str(params["controller"]),
                    "--queue-depth", str(params["queue_depth"]),
                    "--num-queues", str(params["num_queue"]),
                    "--block-size", str(params["block_size"]),
                    "--num-blocks", str(params["num_blocks"]),
                    "--page-size", str(params["page_size"]),
                    "--io-type", str(params["io_type_id"]),
                    "--io-method", str(params["io_method_id"]),
                    "--num-io", str(params["num_io"]),
                    "--ssd-size", "3500", "GB"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, 
                # timeout=180 # 3 minutes
            )
                
            f.write(result.stdout)
            
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
    except Exception as e:
        print(f"Failed to run command: {e}")
        # delete the file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            


def plot_benchmark_results(
    output_dir,
    config_label,
    x_axis='ps',            # Which parameter to plot on the X-axis: 'qd', 'nq', 'bs', 'ps'
    constraints=None,       # Dictionary of constraints to filter on, e.g. {'qd': 1024, 'nq': 1, 'bs': 1}
    image_path=None         # If None, will auto-generate a file name based on x_axis and config_label
):
    """
    Plots bandwidth and throughput for both 'our' and 'bam' benchmarks,
    with an option to specify which parameter is on the X-axis and to filter
    on the other parameters. Also includes SSD specification horizontal lines
    with "Max" annotations.

    Parameters:
    - output_dir   (str): Path to the directory containing benchmark output files.
    - config_label (str): Label for the configuration (e.g., "random-read").
    - x_axis       (str): The parameter to use for the X-axis. One of {'qd', 'nq', 'bs', 'ps'}.
    - constraints  (dict): A dictionary specifying fixed values for other parameters to filter by,
                           e.g. {'qd': 1024, 'nq': 1, 'bs': 1}. Files that do not match
                           these constraints are excluded.
    - image_path   (str): Path where the generated image will be saved. If None, an automatic
                          name is generated based on config_label and x_axis.
    """

    if constraints is None:
        constraints = {}

    # If no image_path is specified, generate one based on the config label and the chosen x_axis
    if image_path is None:
        image_path = f"{config_label}_{x_axis}_plot.png"

    image_path = os.path.join(output_dir, image_path)
    output_dir = os.path.join(output_dir, config_label)

    # Regular expression to extract parameters from filename
    #   Group 1 = our|bam
    #   Group 2 = qd (queue depth)
    #   Group 3 = nq (num queues)
    #   Group 4 = bs (block size in bytes)
    #   Group 5 = nb (not sure, possibly 'num_block' from your example)
    #   Group 6 = ps (page size or something else)
    #   Group 7 = read|write
    #   Group 8 = random|sequential
    #   Group 9 = io _count
    #
    # Adjust or rename groups as needed if your original usage is different.
    #
    # Example filename: our-qd1024-nq1-bs4096-nb10-ps4-read-random-123.txt
    filename_pattern = re.compile(
        r'^(our|bam)-qd(\d+)-nq(\d+)-bs(\d+)-nb(\d+)-ps(\d+)-(read|write)-(random|sequential)-nio\d+\.txt$'
    )

    # Initialize data structures
    data = {
        'our': {},
        'bam': {}
    }

    # This will store the total threads if needed (depending on your usage)
    file_info = {}

    # -------------------------------------------------------------------------
    # 1. Parse files, filter by constraints, and store bandwidth/throughput
    # -------------------------------------------------------------------------
    for filename in os.listdir(output_dir):
        match = filename_pattern.match(filename)
        if not match:
            continue  # Skip files that do not match the pattern

        (
            benchmark_type,
            qd_str,
            nq_str,
            bs_str,
            nb_str,
            ps_str,
            io_type,
            io_method,
        ) = match.groups()
        
        
        # Ensure the file corresponds to the current configuration (read/write + random/sequential)
        if f"{io_method}-{io_type}" != config_label:
            continue
        
        print(f"Processing file: {filename}")

        # Convert to integer
        qd_val = int(qd_str)
        nq_val = int(nq_str)
        bs_val = int(bs_str)
        nb_val = int(nb_str)
        ps_val = int(ps_str)

        # Filter out data that doesn't match constraints
        # e.g. if constraints={"qd":1024, "nq":1, "bs":4096}, we only keep files that match exactly
        # If constraints is empty or a key is absent, we skip that check
        def matches_constraints(param_name, param_value):
            """Return True if constraints do not specify a different value for param_name."""
            return (param_name not in constraints) or (constraints[param_name] == param_value)

        if not (matches_constraints('qd', qd_val) and
                matches_constraints('nq', nq_val) and
                matches_constraints('bs', bs_val) and
                matches_constraints('ps', ps_val)):
            continue

        # The user asked for total threads to be "num_block * block_size" in original code,
        # but that might not be meaningful if we are changing the dimension of the plot.
        # We'll keep it if you still want it:
        # total_threads.append(nb_val * bs_val)
        file_info[filename] = {
            'qd': qd_val,
            'nq': nq_val,
            'bs': bs_val,
            'ps': ps_val,
            'nb': nb_val
        }

        # Decide the x-axis value:
        if x_axis == 'qd':
            x_value = qd_val
        elif x_axis == 'nq':
            x_value = nq_val
        elif x_axis == 'bs':
            x_value = bs_val
        elif x_axis == 'ps':
            x_value = ps_val
        else:
            raise ValueError(f"Invalid x_axis: {x_axis}. Must be one of ['qd', 'nq', 'bs', 'ps'].")

        # Initialize data structure if needed
        if x_value not in data[benchmark_type]:
            data[benchmark_type][x_value] = {'bandwidth': None, 'throughput': None}

        # Read the file and extract bandwidth/throughput
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'r') as f:
            for line in f:
                if "Bandwidth:" in line:
                    try:
                        bandwidth_str = line.split(':', 1)[1].strip().split()[0]
                        data[benchmark_type][x_value]['bandwidth'] = float(bandwidth_str)
                    except (IndexError, ValueError):
                        pass
                elif "Throughput:" in line:
                    try:
                        throughput_str = line.split(':', 1)[1].strip().split()[0]
                        data[benchmark_type][x_value]['throughput'] = float(throughput_str)
                    except (IndexError, ValueError):
                        pass

    # -------------------------------------------------------------------------
    # 2. Compute missing metrics if possible
    #    (e.g., if we have throughput but not bandwidth, or vice versa)
    # -------------------------------------------------------------------------
    for benchmark in ['our', 'bam']:
        for x_val, metrics in data[benchmark].items():
            bw = metrics['bandwidth']
            th = metrics['throughput']
            # If block size is on the x-axis, we can get the BS from x_val, but only if x_axis=='bs'
            # Otherwise, we have no direct way to recover BS from the x_val alone unless we store it.
            #
            # If your original logic to "derive one from the other" only makes sense
            # if 'bs' is known, then we need that. You might want to skip this part for other x-axes.
            #
            # We'll demonstrate the same logic as before, but only if x_axis=='bs':
            if x_axis == 'bs':
                block_size = x_val
                if bw is None and th is not None:
                    # bandwidth = throughput * block_size / (1024^2)
                    data[benchmark][x_val]['bandwidth'] = th * block_size / (1024 * 1024)
                elif th is None and bw is not None and block_size > 0:
                    # throughput = bandwidth * (1024^2) / block_size
                    data[benchmark][x_val]['throughput'] = bw * (1024 * 1024) / block_size

    # -------------------------------------------------------------------------
    # 3. Prepare data for plotting
    # -------------------------------------------------------------------------
    # Collect the sorted X-values that are present for either 'our' or 'bam'
    x_values = sorted(set(data['our'].keys()) | set(data['bam'].keys()))

    our_bandwidth    = [data['our'].get(x_val, {}).get('bandwidth', 0) for x_val in x_values]
    bam_bandwidth    = [data['bam'].get(x_val, {}).get('bandwidth', 0) for x_val in x_values]
    our_throughput   = [data['our'].get(x_val, {}).get('throughput', 0) for x_val in x_values]
    bam_throughput   = [data['bam'].get(x_val, {}).get('throughput', 0) for x_val in x_values]

    # -------------------------------------------------------------------------
    # 4. Create subplots and plot
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))

    # Title can incorporate total_threads if relevant
    # If total_threads is not meaningful for your new approach, you can remove it
    suptitle_str = f'{config_label}'
    
    total_threads = []
    for filename, info in file_info.items():
        total_threads.append(info['bs'] * info['nb'])
    # total_io = total_threads * page_size
    total_io_bytes = []
    for filename, info in file_info.items():
        total_io_bytes.append(info['bs'] * info['nb'] * info['ps'])
    
    suptitle_str += f' - Total IO: {bytes_to_str(total_io_bytes[0])}'
    suptitle_str += f' - Total Threads Per Block: {constraints["bs"]}'
    # if total_threads is not None:
        # suptitle_str += f' - Total Threads: {total_threads}'
    fig.suptitle(suptitle_str)

    # Plot Bandwidth
    axs[0].plot(x_values, our_bandwidth, marker='o', label='Our Benchmark')
    axs[0].plot(x_values, bam_bandwidth, marker='s', label='BAM Benchmark')
    axs[0].set_xlabel(x_axis.upper())  # Label the X-axis with the name of the parameter
    axs[0].set_ylabel('Bandwidth (MB/s)')
    axs[0].set_ylim(bottom=0)
    axs[0].set_xlim(left=0)
    axs[0].grid(True)
    axs[0].legend()

    # Plot Throughput
    axs[1].plot(x_values, our_throughput, marker='o', label='Our Benchmark')
    axs[1].plot(x_values, bam_throughput, marker='s', label='BAM Benchmark')
    axs[1].set_xlabel(x_axis.upper())
    axs[1].set_ylabel('IOPS')
    axs[1].set_ylim(bottom=0)
    axs[1].set_xlim(left=0)
    axs[1].grid(True)
    axs[1].legend()

    # -------------------------------------------------------------------------
    # 5. Add SSD specification lines if desired
    #    (same logic as before, but referencing config_label)
    # -------------------------------------------------------------------------
    # Note: Adjust these values or remove if not applicable for your new approach
    if config_label.startswith("sequential"):
        if "read" in config_label:
            # e.g., 6800 MB/s
            axs[0].axhline(y=6800, color='r', linestyle='--', label='Max Read BW')
            # Put text a bit below that line
            mid_idx = len(x_values) // 2
            if mid_idx < len(x_values):
                axs[0].text(x_values[mid_idx], 6500, 'Max Read BW', color='r',
                            va='center', ha='left', fontsize=12)
        elif "write" in config_label:
            # e.g., 4000 MB/s
            axs[0].axhline(y=4000, color='g', linestyle='--', label='Max Write BW')
            mid_idx = len(x_values) // 2
            if mid_idx < len(x_values):
                axs[0].text(x_values[mid_idx], 3800, 'Max Write BW', color='g',
                            va='center', ha='left', fontsize=12)
    elif config_label.startswith("random"):
        if "read" in config_label:
            # e.g., 1,000,000 IOPS
            axs[1].axhline(y=1_000_000, color='r', linestyle='--', label='Max Read IOPS')
            mid_idx = len(x_values) // 2
            if mid_idx < len(x_values):
                axs[1].text(x_values[mid_idx], 900_000, 'Max Read IOPS', color='r',
                            va='center', ha='left', fontsize=12)
        elif "write" in config_label:
            # e.g., 180,000 IOPS
            axs[1].axhline(y=180_000, color='g', linestyle='--', label='Max Write IOPS')
            mid_idx = len(x_values) // 2
            if mid_idx < len(x_values):
                axs[1].text(x_values[mid_idx], 150_000, 'Max Write IOPS', color='g',
                            va='center', ha='left', fontsize=12)

    # -------------------------------------------------------------------------
    # 6. Adjust Y-limits if you want to ensure the max specs are visible
    # -------------------------------------------------------------------------
    if config_label.startswith("sequential"):
        if "read" in config_label:
            lim = max(our_bandwidth + bam_bandwidth + [6800])
            axs[0].set_ylim(top=lim + lim * 0.1)
        elif "write" in config_label:
            lim = max(our_bandwidth + bam_bandwidth + [4000])
            axs[0].set_ylim(top=lim + lim * 0.1)
    elif config_label.startswith("random"):
        if "read" in config_label:
            lim = max(our_throughput + bam_throughput + [1_000_000])
            axs[1].set_ylim(top=lim + lim * 0.1)
        elif "write" in config_label:
            lim = max(our_throughput + bam_throughput + [180_000])
            axs[1].set_ylim(top=lim + lim * 0.1)

    # Update legends with new lines
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    # -------------------------------------------------------------------------
    # 7. Save the figure
    # -------------------------------------------------------------------------
    plt.tight_layout()
    # plt.savefig(image_path, transparent=True)
    plt.savefig(image_path)
    plt.close()
    print(f"Plot saved to {image_path}")


def plot_latencies(output_dir, config_label, image_path, num_io, sample_size=1):
    """
    Parses latency data from files in the output directory and plots them.

    Args:
        output_dir (str): Path to the directory containing latency files.
        config_label (str): Label for the plot title.
        image_path (str): Path to save the generated plot image.
    """
    # Compile the regular expression once
    filename_pattern = re.compile(
        r'^(?:our|bam)-qd\d+-nq\d+-bs(\d+)-nb(\d+)-ps\d+-(?:read|write)-(?:random|sequential)-(\d+)_clock\.txt$'
    )

    # Use defaultdict to automatically handle missing keys
    latencies = defaultdict(list)

    output_path = Path(output_dir)
    if not output_path.is_dir():
        raise ValueError(f"Output directory '{output_dir}' does not exist or is not a directory.")

    # Iterate over all files in the output directory
    for file_path in output_path.iterdir():
        if not file_path.is_file():
            continue  # Skip non-file entries

        match = filename_pattern.match(file_path.name)
        if not match:
            continue  # Skip files that do not match the pattern
        

        block_size, num_block, num_io_match = match.groups()

        if int(num_io_match) != num_io:
            continue      
        
        
        label = f"bs: {block_size} nb: {num_block}"

        try:
            with file_path.open('r') as f:
                lines = f.readlines()

            # Find the index of the line containing "Throughput:"
            throughput_index = next(i for i, line in enumerate(lines) if "Throughput:" in line) + 1
        except (StopIteration, IOError):
            continue  # Skip files without "Throughput:" or unreadable files

        # Extract latency values until an empty line or end of file
        for line in lines[throughput_index:]:
            stripped = line.strip()
            if not stripped:
                break  # Stop if an empty line is encountered
            try:
                # line = line.split(" ")
                stripped = stripped.split(" ")
                # latency = int(stripped)
                latencies[label] += [int(x) for x in stripped]
            except ValueError:
                continue  # Ignore lines that cannot be converted to integers
            
        print(f"Latency data extracted from {file_path}")

    if not latencies:
        print("No latency data found to plot.")
        return

    # Sort labels by block size in descending order
    sorted_labels = sorted(
        latencies.keys(),
        key=lambda x: int(x.split()[1]),
        reverse=True
    )

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'{config_label} - Latency')

    # Plot each latency series
    for label in sorted_labels:
        data = np.array(latencies[label])

        # take a random sample of 10% of the data
        sample = np.random.choice(data, int(len(data) * sample_size), replace=False)
        
        # scatter plot
        ax.scatter(np.arange(len(sample)), sample, label=label, linestyle='None', marker='o', s=1)
        
        # ax.scatter(data, label=label, linestyle='None', marker='o')  # plot as points
        # ax.scatter(np.arange(len(data)), data, label=label, linestyle='None', marker='o')  # plot as points

    # Set axis labels
    ax.set_xlabel('Thread ID')
    ax.set_ylabel('Latency (Clock Cycles)')

    # Configure the legend
    legend = ax.legend(fontsize=12, loc='best', title="Configurations")
    for handle in legend.legend_handles:
        handle.set_linewidth(2.0)  # Thicker lines in the legend for better visibility

    # Add grid and adjust layout
    ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the suptitle

    # Save and close the plot to free up resources
    plt.savefig(image_path, transparent=True)
    print(f"Plot saved to {image_path}")
    plt.close(fig)
    
    
    
    

# if main
if __name__ == "__main__":
    io_types_to_exec = ["write", "read"]

    io_methods_to_exec = ["random"]
    
    page_sizes = [512, 1024, 2048, 4096, 8192, 16384]

    # Define the range of parameters
    queue_depths = 1024
    num_queues = 1
    block_sizes = [1]
    # Dynamically generate num_blocks based on block_sizes
    num_io = 1

    # data to operate on
    data = 16 * GIGABYTE
    
    for io_type in io_types_to_exec:
        for io_method in io_methods_to_exec:
            for page_size in page_sizes:
                # make folder
                output_dir = f"{io_method}-{io_type}"
                output_dir = os.path.join(file_dir, output_dir)
                os.makedirs(output_dir, exist_ok=True)
                # generate block sizes based on page size and queue depth
                num_ssd_pages = int(data / page_size)
                num_blocks = [int(num_ssd_pages / block_size) for block_size in block_sizes]

                # enumerate through num blocks
                for block_size, num_block in zip(block_sizes, num_blocks):
                    params = {
                        "device": 0,
                        "controller": 0,
                        "queue_depth": queue_depths,
                        "num_queue": num_queues,
                        "block_size": block_size,
                        "num_blocks": num_block,
                        "page_size": page_size,
                        "io_type": io_type,
                        "io_method": io_method,
                        "num_io": num_io
                    }

                    our_benchmark(our_exec_path, params, output_dir)
                    # our_benchmark(our_exec_path, params, output_dir, clock=True)
                    bam_benchmark(bam_exec_path, params, output_dir)
            
    plot_benchmark_results(file_dir, "random-read", x_axis='ps', constraints={"bs": 1, "qd": 1024, "nq": 1})
    plot_benchmark_results(file_dir, "random-write", x_axis='ps', constraints={"bs": 1, "qd": 1024, "nq": 1})
    
        
        
    