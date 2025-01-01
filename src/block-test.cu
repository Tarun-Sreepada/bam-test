#include "settings.cuh" // Contains definition of `Settings`, `parse_arguments()`, etc.
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstdlib>

// Define the error checking macro
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

// Kernel Definitions
__global__ void rand_read_kernel(Controller **controllers,
                                 page_cache_d_t *device_page_cache,
                                 uint64_t page_size,
                                 uint64_t *offsets, uint64_t num_offsets,
                                 uint64_t num_requests)
{
    // Thread and warp identification
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t laneid = lane_id();

    uint64_t ctrl;
    uint64_t queue;

    if (laneid == 0)
    {
        ctrl = device_page_cache->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % 1;
        queue = tid % (controllers[ctrl]->n_qps);
    }

    // Broadcast the controller and queue to all threads in the warp
    ctrl = __shfl_sync(0xffffffff, ctrl, 0);
    queue = __shfl_sync(0xffffffff, queue, 0);

    // 3) Convert logical block size to physical block size
    uint64_t block_size_log = controllers[ctrl]->d_qps[queue].block_size_log;
    uint64_t block_size = (1 << block_size_log); // e.g., 1 << 9 = 512 bytes
    uint64_t n_blocks = page_size / block_size;
    uint64_t index_start = tid * num_requests;

#ifdef CLOCK
    uint64_t start = clock64();
#endif

    // Each thread will do multiple read requests
    for (uint64_t i = 0; i < num_requests; i++)
    {

        // 2) Retrieve the byte offset from offsets array with proper 64-bit conversion
        uint64_t offset_bytes = offsets[index_start++] * page_size;

        // If offset_bytes is an absolute byte offset on the device,
        // we convert it to a block offset by dividing by block_size.
        // Similarly, we figure out how many blocks we need to read.
        uint64_t start_block = offset_bytes / block_size;

        // 4) Issue the read request
        read_data(device_page_cache,                  // page cache
                  &(controllers[ctrl]->d_qps[queue]), // queue to use
                  start_block,                        // from where to start reading
                  n_blocks,                           // how many blocks to read
                  tid);                               // to which thread to write the data

// Conditionally record the clock using preprocessor directives
#ifdef CLOCK
        uint64_t end = clock64();
        offsets[index_start - 1] = end - start;
        start = end;
#endif
    }
}

__global__ void rand_write_kernel(Controller **controllers,
                                  page_cache_d_t *device_page_cache,
                                  uint64_t page_size,
                                  uint64_t *offsets, uint64_t num_offsets,
                                  uint64_t num_requests)
{
    // Thread and warp identification
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t laneid = lane_id();

    uint64_t ctrl;
    uint64_t queue;

    if (laneid == 0)
    {
        ctrl = device_page_cache->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % 1;
        queue = tid % (controllers[ctrl]->n_qps);
    }

    // Broadcast the controller and queue to all threads in the warp
    ctrl = __shfl_sync(0xffffffff, ctrl, 0);
    queue = __shfl_sync(0xffffffff, queue, 0);

    // 3) Convert logical block size to physical block size
    uint64_t block_size_log = controllers[ctrl]->d_qps[queue].block_size_log;
    uint64_t block_size = (1 << block_size_log); // e.g., 1 << 9 = 512 bytes
    uint64_t n_blocks = page_size / block_size;
    uint64_t index_start = tid * num_requests;

#ifdef CLOCK
    uint64_t start = clock64();
#endif

    // Each thread will do multiple read requests
    for (uint64_t i = 0; i < num_requests; i++)
    {

        // 2) Retrieve the byte offset from offsets array with proper 64-bit conversion
        uint64_t offset_bytes = offsets[index_start++] * page_size;

        // If offset_bytes is an absolute byte offset on the device,
        // we convert it to a block offset by dividing by block_size.
        // Similarly, we figure out how many blocks we need to read.
        uint64_t start_block = offset_bytes / block_size;

        // 4) Issue the read request
        write_data(device_page_cache,                  // page cache
                   &(controllers[ctrl]->d_qps[queue]), // queue to use
                   start_block,                        // from where to start reading
                   n_blocks,                           // how many blocks to read
                   tid);                               // to which thread to write the data

// Conditionally record the clock using preprocessor directives
#ifdef CLOCK
        uint64_t end = clock64();
        offsets[index_start - 1] = end - start;
        start = end;
#endif
    }
}

__global__ void seq_write_kernel(Controller **controllers,
                                 page_cache_d_t *device_page_cache,
                                 uint64_t *offsets,
                                 uint64_t page_size, uint64_t num_requests)
{
    // Thread and warp identification
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_threads = gridDim.x * blockDim.x;
    uint64_t laneid = lane_id();
    uint64_t smid = get_smid();

    uint64_t ctrl;
    uint64_t queue;

    if (laneid == 0)
    {
        ctrl = device_page_cache->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % 1;
        queue = smid % (controllers[ctrl]->n_qps);
    }

    // Broadcast the controller and queue to all threads in the warp
    ctrl = __shfl_sync(0xffffffff, ctrl, 0);
    queue = __shfl_sync(0xffffffff, queue, 0);

    // 3) Convert logical block size to physical block size
    uint64_t block_size_log = controllers[ctrl]->d_qps[queue].block_size_log;
    uint64_t block_size = (1 << block_size_log); // e.g., 1 << 9 = 512 bytes
    uint64_t n_blocks = page_size >> block_size_log;
    uint64_t index_start = tid;

#ifdef CLOCK
    uint64_t start = clock64();
#endif

    // Each thread will do multiple read requests
    for (uint64_t i = 0; i < num_requests; i++)
    {

        // 2) Retrieve the byte offset from offsets array with proper 64-bit conversion
        uint64_t offset_bytes = index_start * page_size;

        // If offset_bytes is an absolute byte offset on the device,
        // we convert it to a block offset by dividing by block_size.
        // Similarly, we figure out how many blocks we need to read.
        // uint64_t start_block = offset_bytes / block_size;
        uint64_t start_block = offset_bytes >> block_size_log;

        // 4) Issue the read request
        write_data(device_page_cache,                  // page cache
                   &(controllers[ctrl]->d_qps[queue]), // queue to use
                   start_block,                        // from where to start reading
                   n_blocks,                           // how many blocks to read
                   tid);                               // to which thread to write the data

// Conditionally record the clock using preprocessor directives
#ifdef CLOCK
        uint64_t end = clock64();
        offsets[index_start] = end - start;
        start = end;
#endif

        index_start += total_threads;
    }
}

__global__ void seq_read_kernel(Controller **controllers,
                                page_cache_d_t *device_page_cache,
                                uint64_t *offsets,
                                uint64_t page_size, uint64_t num_requests)
{
    // Thread and warp identification
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_threads = gridDim.x * blockDim.x;
    uint64_t laneid = lane_id();
    uint32_t smid = get_smid();

    uint64_t ctrl;
    uint64_t queue;

    if (laneid == 0)
    {
        ctrl = device_page_cache->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % 1;
        queue = smid % (controllers[ctrl]->n_qps);
    }

    // Broadcast the controller and queue to all threads in the warp
    ctrl = __shfl_sync(0xffffffff, ctrl, 0);
    queue = __shfl_sync(0xffffffff, queue, 0);

    // 3) Convert logical block size to physical block size
    uint64_t block_size_log = controllers[ctrl]->d_qps[queue].block_size_log;
    uint64_t block_size = (1 << block_size_log); // e.g., 1 << 9 = 512 bytes
    uint64_t n_blocks = page_size / block_size;
    uint64_t index_start = tid;

#ifdef CLOCK
    uint64_t start = clock64();
#endif

    // Each thread will do multiple read requests
    for (uint64_t i = 0; i < num_requests; i++)
    {

        // 2) Retrieve the byte offset from offsets array with proper 64-bit conversion
        uint64_t offset_bytes = index_start * page_size;

        // If offset_bytes is an absolute byte offset on the device,
        // we convert it to a block offset by dividing by block_size.
        // Similarly, we figure out how many blocks we need to read.
        uint64_t start_block = offset_bytes / block_size;

        // 4) Issue the read request
        read_data(device_page_cache,                  // page cache
                  &(controllers[ctrl]->d_qps[queue]), // queue to use
                  start_block,                        // from where to start reading
                  n_blocks,                           // how many blocks to read
                  tid);                               // to which thread to write the data

// Conditionally record the clock using preprocessor directives
#ifdef CLOCK
        uint64_t end = clock64();
        offsets[index_start] = end - start;
        start = end;
#endif

        index_start += total_threads;
    }
}

int main(int argc, char **argv)
{
    cudaError_t err;
#ifdef CLOCK
        std::cout << "CLOCK ENABLED" << std::endl;
#endif
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

    // 2. Create controller(s)
    //    We assume 1 controller here, but you could do more
    std::vector<Controller *> controllers;
    controllers.push_back(
        new Controller(settings.selected_controller_path.c_str(),
                       1,               /*nvmNamespace=*/
                       settings.device, /*cudaDevice=*/
                       settings.queue_depth,
                       settings.num_queues));

    // 3. Initialize Page Cache
    page_cache_t host_page_cache(settings.page_size,
                                 settings.num_blocks * settings.block_size, // only 1 page per thread
                                 settings.device,
                                 controllers[0][0], // we only using 1 controller
                                 /*max range*/ 64,  // kept as 64 for now
                                 controllers);

    // Get the device-side page cache pointer
    page_cache_d_t *device_page_cache =
        reinterpret_cast<page_cache_d_t *>(host_page_cache.d_pc_ptr);

    // 4. Generate I/O Offsets
    uint64_t num_off_requests = settings.num_io_requests * settings.num_blocks * settings.block_size;

    std::vector<uint64_t> offsets;
    offsets.reserve(num_off_requests);
    uint64_t *d_offsets = nullptr;

    if (settings.io_method == RANDOM)
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(
            0, (settings.ssd_pages - 1));

        for (int i = 0; i < num_off_requests; ++i)
        {
            // Pick a random page, convert to a byte offset
            uint64_t random_page = dis(gen);
            offsets.push_back(random_page);
        }

        // 5. Allocate and copy offsets array to GPU
        uint64_t offsets_size = num_off_requests * sizeof(uint64_t);

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
    }
    else
    {
        err = cudaMalloc(&d_offsets, num_off_requests * sizeof(uint64_t));
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to allocate device memory for offsets: "
                      << cudaGetErrorString(err) << std::endl;
            return EXIT_FAILURE;
        }

        // Initialize the offsets array
        err = cudaMemset(d_offsets, 0, num_off_requests * sizeof(uint64_t));
    }

    // 6. Prepare for kernel launch (timing, etc.)

    uint64_t num_blocks = settings.num_blocks;
    uint64_t block_size = settings.block_size;
    uint64_t num_io_reqs = settings.num_io_requests; // how many requests each thread will handle

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    // 7. Launch read or write kernel

    if (settings.io_method == RANDOM)
    {
        if (settings.io_type == READ)
        {
            rand_read_kernel<<<num_blocks, block_size>>>(
                host_page_cache.pdt.d_ctrls,
                device_page_cache,
                settings.page_size,
                d_offsets, num_off_requests,
                num_io_reqs);
        }
        else
        {
            rand_write_kernel<<<num_blocks, block_size>>>(
                host_page_cache.pdt.d_ctrls,
                device_page_cache,
                settings.page_size,
                d_offsets, num_off_requests,
                num_io_reqs);
        }
    }
    else
    {
        if (settings.io_type == READ)
        {
            seq_read_kernel<<<num_blocks, block_size>>>(
                host_page_cache.pdt.d_ctrls,
                device_page_cache, d_offsets,
                settings.page_size,
                num_io_reqs);
        }
        else
        {
            seq_write_kernel<<<num_blocks, block_size>>>(
                host_page_cache.pdt.d_ctrls,
                device_page_cache, d_offsets,
                settings.page_size,
                num_io_reqs);
        }
    }

    gpuErrchk(cudaPeekAtLastError());

    // Record the stop event
    gpuErrchk(cudaEventRecord(stop, 0));
    // Wait for stop event to complete
    gpuErrchk(cudaEventSynchronize(stop));

    // 8. Measure performance and print results
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
    uint64_t data_transferred = num_io_reqs * settings.page_size * num_blocks * block_size;

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

// 9. Conditionally Handle Clock Data
#ifdef CLOCK
    {
        // Calculate the number of clock records
        // Allocate host memory to receive clock data
        std::vector<uint64_t> clocks(num_off_requests);

        // Copy clock data from device to host
        err = cudaMemcpy(clocks.data(), d_offsets, num_off_requests * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to copy clock data to host: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_offsets);
            return EXIT_FAILURE;
        }

        // Print clock data
        uint64_t num_threads = num_blocks * block_size;
        
        uint64_t conuter = 0;
        for (size_t i = 0; i < num_threads; i++)
        {
            for (size_t j = 0; j < num_io_reqs; j++)
            {
                std::cout << clocks[conuter++] << " ";
            }
            std::cout << std::endl;

        }
        std::cout << std::endl;
    }
#endif

    // 10. Clean-up
    cudaFree(d_offsets);

    // Free controllers
    for (auto *c : controllers)
        delete c;

    return EXIT_SUCCESS;
}
