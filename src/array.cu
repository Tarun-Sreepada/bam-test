#include "settings.cuh"



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

    //---------------------------------------------------------

    page_cache_t host_page_cache(settings.page_size,
                                 settings.num_blocks * settings.block_size, // each thread has 1 page
                                 settings.device,
                                 controllers[0][0], // we only using 1 controller
                                 /*max range*/ 64,  // i dont know why its 64 but im keeping it 64 for now
                                 controllers);

    // Create the page cache on the device
    page_cache_d_t *device_page_cache =
        reinterpret_cast<page_cache_d_t *>(host_page_cache.d_pc_ptr);

    //---------------------------------------------------------
    // create range
    // range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, 
        // uint64_t pso, uint64_t p_size, page_cache_t *c_h, uint32_t cudaDevice, data_dist_t dist = REPLICATE);
    // index_start, number of elements, page_start, number of pages, page_start offset, page size, page cache, cuda device
    // range_t<uint64_t> host_range(1ULL, settings.num_blocks * settings.block_size,

    // array_t<uint64_t> host_range(1ULL, settings.num_blocks * settings.block_size,
    //                              0ULL, settings.num_blocks * settings.block_size,
    //                              0ULL, settings.page_size,
    //                              &host_page_cache, settings.device);



    for (auto *c : controllers)
        delete c;

    return EXIT_SUCCESS;

}