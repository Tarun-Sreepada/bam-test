#include "settings.cuh"            // Contains definition of `Settings`, `parse_arguments()`, etc.
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstdlib>

// Define the error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


// Kernel Definitions
__global__ void read_kernel(Controller **controllers,
                            page_cache_d_t *device_page_cache,
                            uint32_t page_size,
                            uint32_t *offsets, uint32_t num_offsets,
                            int num_requests)
{
    // Thread and warp identification
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = blockIdx.x;
    int laneid = lane_id();
    int smid = get_smid();

    uint32_t ctrl;
    uint32_t queue;

    if (laneid == 0)
    {
        ctrl = device_page_cache->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % 1;
        queue = tid % (controllers[ctrl]->n_qps);
    }

    // Broadcast the controller and queue to all threads in the warp    
    ctrl = __shfl_sync(0xffffffff, ctrl, 0);
    queue = __shfl_sync(0xffffffff, queue, 0);

    // Each thread will do multiple read requests
    for (int i = 0; i < num_requests; i++)
    {
        // 1) Calculate which offset index this thread should use
        int index_into_offsets = (tid * num_requests + i);

        // 2) Retrieve the byte offset from offsets array with proper 64-bit conversion
        uint64_t offset_bytes = static_cast<uint64_t>(offsets[index_into_offsets]) * static_cast<uint64_t>(page_size);

        // 3) Convert logical block size to physical block size
        int block_size_log = controllers[ctrl]->d_qps[queue].block_size_log;
        int block_size     = (1 << block_size_log);   // e.g., 1 << 9 = 512 bytes

        // If offset_bytes is an absolute byte offset on the device,
        // we convert it to a block offset by dividing by block_size.
        // Similarly, we figure out how many blocks we need to read.
        int start_block = offset_bytes / block_size;
        int n_blocks    = page_size / block_size;  // how many blocks fit into 'page_size' bytes

        // 4) Issue the read request
        read_data(device_page_cache, // page cache
                  &(controllers[ctrl]->d_qps[queue]), // queue to use
                  start_block, // from where to start reading
                  n_blocks, // how many blocks to read
                  tid); // to which thread to write the data

        // Conditionally record the clock using preprocessor directives
        #ifdef CLOCK
        offsets[index_into_offsets] = clock();
        #endif
    }
}

__global__ void write_kernel(Controller **controllers,
                             page_cache_d_t *device_page_cache,
                             uint32_t page_size,
                             uint32_t *offsets, uint32_t num_offsets,
                             int num_requests)
{
    // Implement the write logic similarly to read_kernel
    // Include clock recording if CLOCK is defined
    // For brevity, implementation is omitted
}

int main(int argc, char **argv)
{
    cudaError_t err;

    //---------------------------------------------------------
    // 1. Parse and print settings
    //---------------------------------------------------------
    Settings settings = parse_arguments(argc, argv);
    print_settings(settings);

    // Select the GPU device
    err = cudaSetDevice(settings.device);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    //---------------------------------------------------------
    // 2. Create controller(s)
    //    We assume 1 controller here, but you could do more
    //---------------------------------------------------------
    std::vector<Controller *> controllers;
    controllers.push_back(
        new Controller(settings.selected_controller_path.c_str(),
                       1, /*nvmNamespace=*/
                       settings.device, /*cudaDevice=*/
                       settings.queue_depth,
                       settings.num_queues));

    //---------------------------------------------------------
    // 3. Initialize Page Cache
    //---------------------------------------------------------
    page_cache_t host_page_cache(settings.page_size,
                                 settings.num_blocks * settings.block_size,
                                 settings.device,
                                 controllers[0][0], // we only using 1 controller
                                 /*max range*/ 64,  // kept as 64 for now
                                 controllers);

    // Get the device-side page cache pointer
    page_cache_d_t *device_page_cache =
        reinterpret_cast<page_cache_d_t *>(host_page_cache.d_pc_ptr);

    // 4. Generate I/O Offsets
    uint64_t ssd_size_bytes = 1024ULL * 1024 * 1024 * 1024; // 1 TB
    int num_off_requests    = settings.num_io_requests * settings.num_blocks * settings.block_size;

    std::vector<uint32_t> offsets;
    offsets.reserve(num_off_requests);

    if (settings.io_method == SEQUENTIAL)
    {
        // Create offsets in a sequential manner
        for (int i = 0; i < num_off_requests; ++i)
        {
            offsets.push_back(static_cast<uint32_t>(i));
        }
    }
    else // RANDOM
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(
            0, (ssd_size_bytes / settings.page_size) - 1);

        for (int i = 0; i < num_off_requests; ++i)
        {
            // Pick a random page, convert to a byte offset
            uint64_t random_page = dis(gen);
            offsets.push_back(static_cast<uint32_t>(random_page));
        }
    }

    //---------------------------------------------------------
    // 5. Allocate and copy offsets array to GPU
    //---------------------------------------------------------
    uint32_t *d_offsets = nullptr;
    size_t offsets_size = num_off_requests * sizeof(uint32_t);

    err = cudaMalloc(&d_offsets, offsets_size);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory for offsets: "
                  << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(d_offsets, offsets.data(), offsets_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy offsets to device: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_offsets);
        return EXIT_FAILURE;
    }

    //---------------------------------------------------------
    // 6. Prepare for kernel launch (timing, etc.)
    //---------------------------------------------------------
    
    int num_blocks  = settings.num_blocks;
    int block_size  = settings.block_size;
    int num_io_reqs = settings.num_io_requests; // how many requests each thread will handle

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    //---------------------------------------------------------
    // 7. Launch read or write kernel
    //    (Here we only show read_kernel. You would define
    //     write_kernel similarly.)
    //---------------------------------------------------------
    
    if (settings.io_type == READ)
    {
        read_kernel<<<num_blocks, block_size>>>(
            host_page_cache.pdt.d_ctrls,
            device_page_cache,
            settings.page_size,
            d_offsets, num_off_requests,
            num_io_reqs);
    }
    else
    {
        write_kernel<<<num_blocks, block_size>>>(
            host_page_cache.pdt.d_ctrls,
            device_page_cache,
            settings.page_size,
            d_offsets, num_off_requests,
            num_io_reqs);
    }

    gpuErrchk(cudaPeekAtLastError());

    // Record the stop event
    gpuErrchk(cudaEventRecord(stop, 0));
    // Wait for stop event to complete
    gpuErrchk(cudaEventSynchronize(stop));

    //---------------------------------------------------------
    // 8. Measure performance and print results
    //---------------------------------------------------------
    float elapsed_time_ms = 0.0f;
    err = cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get elapsed time: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_offsets);
        return EXIT_FAILURE;
    }

    // Destroy timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Amount of data read/written
    // = (#threads * #requests/thread * page_size)
    // but it depends on how you structure the code. For simplicity:
    size_t data_transferred = static_cast<size_t>(num_io_reqs) * settings.page_size * num_blocks * block_size;

    // Convert to MB
    double data_mb = static_cast<double>(data_transferred) / (1024.0 * 1024.0);

    // Throughput in MB/s
    double throughput = data_mb / (elapsed_time_ms / 1000.0);

    // IOPS
    double total_io = static_cast<double>(num_io_reqs) * num_blocks * block_size;
    double iops = total_io / (elapsed_time_ms / 1000.0);

    // Print results
    std::cout << std::dec;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Kernel execution time: " << elapsed_time_ms << " ms\n";
    std::cout << "Data transferred:      " << data_mb << " MB\n";
    std::cout << "Bandwidth:             " << throughput << " MB/s\n";
    std::cout << "Throughput:            " << iops << " IOPS\n";

    //---------------------------------------------------------
    // 9. Conditionally Handle Clock Data
    //---------------------------------------------------------
    #ifdef CLOCK
    {
        // Calculate the number of clock records
        size_t num_clocks = static_cast<size_t>(num_blocks) * block_size * num_io_reqs;

        // Allocate host memory to receive clock data
        std::vector<uint32_t> clocks(num_clocks);

        // Copy clock data from device to host
        err = cudaMemcpy(clocks.data(), d_offsets, num_off_requests * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to copy clock data to host: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_offsets);
            return EXIT_FAILURE;
        }

        // Print clock data
        for (size_t i = 0; i < num_clocks; i++)
        {
            if (i % num_io_reqs == 0)
                std::cout << std::endl;

            std::cout << clocks[i] << " ";
        }
        std::cout << std::endl;
    }
    #endif

    //---------------------------------------------------------
    // 10. Clean-up
    //---------------------------------------------------------
    cudaFree(d_offsets);

    // Free controllers
    for (auto *c : controllers)
        delete c;

    return EXIT_SUCCESS;
}
