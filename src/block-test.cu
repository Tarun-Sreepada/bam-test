#include "settings.cuh"            // Contains definition of `Settings`, `parse_arguments()`, etc.

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


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

    // broadcast the controller and queue to all threads in the warp    
    ctrl = __shfl_sync(0xffffffff, ctrl, 0);
    queue = __shfl_sync(0xffffffff, queue, 0);


    // Each thread will do multiple read requests
    // Using the 'num_requests' parameter
    for (int i = 0; i < num_requests; i++)
    {

        //---------------------------------------------------------
        // 1) Calculate which offset index this thread should use
        //---------------------------------------------------------
        // This is a simple example that tries to vary the offset
        // used by each iteration, but you can pick your approach.
        int index_into_offsets = (tid * num_requests + i) % (num_requests);

        // 2) Retrieve the byte offset from offsets array
        uint32_t offset_bytes = offsets[index_into_offsets] * page_size;

        //---------------------------------------------------------
        // 3) Convert the offset into "start block" and "n_blocks"
        //    using the block_size_log from the queue
        //---------------------------------------------------------
        int block_size_log = controllers[ctrl]->d_qps[queue].block_size_log;
        int block_size     = (1 << block_size_log);   // e.g., 1 << 9 = 512 bytes

        // If offset_bytes is an absolute byte offset on the device,
        // we convert it to a block offset by dividing by block_size.
        // Similarly, we figure out how many blocks we need to read.
        int start_block = offset_bytes / block_size;
        int n_blocks    = page_size / block_size;  // how many blocks fit into 'page_size' bytes

        // 4) Issue the read request
        //    read_data( page_cache_d_t*, queue_ptr, start_block, n_blocks, thread_id );
        // read_data(page_cache_d_t *pc, QueuePair *qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry)
        read_data(device_page_cache,
                  &(controllers[ctrl]->d_qps[queue]),
                  start_block, // from where to start reading
                  n_blocks, // how many blocks to read
                  tid); // to which thread to write the data
    }
    
}


__global__ void write_kernel(Controller **controllers,
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

    // broadcast the controller and queue to all threads in the warp    
    ctrl = __shfl_sync(0xffffffff, ctrl, 0);
    queue = __shfl_sync(0xffffffff, queue, 0);


    // Each thread will do multiple read requests
    // Using the 'num_requests' parameter
    for (int i = 0; i < num_requests; i++)
    {

        //---------------------------------------------------------
        // 1) Calculate which offset index this thread should use
        //---------------------------------------------------------
        // This is a simple example that tries to vary the offset
        // used by each iteration, but you can pick your approach.
        int index_into_offsets = (tid + i) % (num_offsets);

        // 2) Retrieve the byte offset from offsets array
        uint32_t offset_bytes = offsets[index_into_offsets] * page_size;

        //---------------------------------------------------------
        // 3) Convert the offset into "start block" and "n_blocks"
        //    using the block_size_log from the queue
        //---------------------------------------------------------
        int block_size_log = controllers[ctrl]->d_qps[queue].block_size_log;
        int block_size     = (1 << block_size_log);   // e.g., 1 << 9 = 512 bytes

        // If offset_bytes is an absolute byte offset on the device,
        // we convert it to a block offset by dividing by block_size.
        // Similarly, we figure out how many blocks we need to read.
        int start_block = offset_bytes / block_size;
        int n_blocks    = page_size / block_size;  // how many blocks fit into 'page_size' bytes

        // 4) Issue the read request
        //    read_data( page_cache_d_t*, queue_ptr, start_block, n_blocks, thread_id );
        write_data(device_page_cache,
                  &(controllers[ctrl]->d_qps[queue]),
                  start_block,
                  n_blocks,
                  tid);
    }
    
}

//-------------------------------------------------------------------
// main()
//  - Parses command-line arguments into a Settings object
//  - Initializes GPU device and data structures
//  - Allocates/copies offsets array
//  - Launches the kernel
//  - Measures performance and prints results
//-------------------------------------------------------------------
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
                       // I don't know what it means but for Samsung drives they used 1 so im going to assume its 1 for micron because intel optane was 0
                       // btw 7450SSD supports upto 128 namespaces from the product brief so we can set it to 128
                       settings.device, /*cudaDevice=*/
                       settings.queue_depth,
                       settings.num_queues));
    /*
        Page Cache:
            Copies the controllers created in the main function to the device
            Ref:
                for (size_t k = 0; k < pdt.n_ctrls; k++)
                    cuda_err_chk(cudaMemcpy(pdt.d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
            
        CreateBuffer:
            Allocs memory on device

        cache_pages_buf:
            Actually allocates the cache pages on the device

        this->pages_dma : 
            create DMA region to hold the data. 1UL << 16 is 64KB of data
                Create DMA mapping descriptor from CUDA device pointer using the kernel
                module. This function is similar to nvm_dma_map_host, except the memory
                pointer must be a valid CUDA device pointer (see manual for
                cudaGetPointerAttributes).
        
        cache_page_t *tps = new cache_page_t[np]
            Allocates memory for the cache pages on the device. A page cache is essentially bunch of smaller 64KB pages (most likely)

     */

    //---------------------------------------------------------
    // 3. Initialize Page Cache
    //    page_cache_t is a wrapper managing GPU buffers
    //---------------------------------------------------------
    page_cache_t host_page_cache(settings.page_size,
                                 settings.num_blocks * settings.block_size,
                                 settings.device,
                                 controllers[0][0], // we only using 1 controller
                                 /*max range*/ 64,  // i dont know why its 64 but im keeping it 64 for now
                                 controllers);


    // Get the device-side page cache pointer
    page_cache_d_t *device_page_cache =
        reinterpret_cast<page_cache_d_t *>(host_page_cache.d_pc_ptr);

    //---------------------------------------------------------
    // 4. Generate I/O Offsets
    //    We'll fill a host vector with either sequential or
    //    random offsets in bytes.
    //---------------------------------------------------------
    uint64_t ssd_size_bytes = 1024ULL * 1024 * 1024 * 1024; // 1 TB, for example
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
    
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    //---------------------------------------------------------
    // 7. Launch read or write kernel
    //    (Here we only show read_kernel. You would define
    //     write_kernel similarly.)
    //---------------------------------------------------------
    int num_blocks  = settings.num_blocks;
    int block_size  = settings.block_size;
    int num_io_reqs = settings.num_io_requests; // how many requests each thread will handle

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
    gpuErrchk(cudaDeviceSynchronize());

    // Record the stop event
    gpuErrchk(cudaEventRecord(stop));

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

    // iops
    double total_io = num_io_reqs * num_blocks * block_size;
    double iops = total_io / (elapsed_time_ms / 1000.0);

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Kernel execution time: " << elapsed_time_ms << " ms\n";
    std::cout << "Data transferred:      " << data_mb << " MB\n";
    std::cout << "Bandwidth:             " << throughput << " MB/s\n";
    std::cout << "Throughput:            " << iops << " IOPS\n";

    //---------------------------------------------------------
    // 9. Clean-up
    //---------------------------------------------------------
    cudaFree(d_offsets);

    // Free controllers
    for (auto *c : controllers)
        delete c;

    return EXIT_SUCCESS;
}


//  sudo ./bin/nvm-io-test-bench --device 0 --controller 0 --queue-depth 2 --num-queues 1 --block-size 1 --num-blocks 1 --page-size 4096 --io-type 0 --io-method 1 --num-io 100000