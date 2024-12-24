#pragma once
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <functional>
#include <cctype>
#include <getopt.h>
#include <limits>
#include <random>

// Project-specific headers (bam/include)
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <ctrl.h>
#include <buffer.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA_ERROR(err)                                                                           \
    if (err != cudaSuccess)                                                                             \
    {                                                                                                   \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        cudaFree(d_offsets); /* Adjust based on allocated resources */                                  \
        return EXIT_FAILURE;                                                                            \
    }

// Enumeration for IO Types
enum IOType
{
    READ = 0,
    WRITE = 1
};

// Enumeration for IO Methods
enum IOMethod
{
    SEQUENTIAL = 0,
    RANDOM = 1
};

struct Settings
{
    // CUDA Parameters
    int device = -1;
    size_t controller_index = 0;
    uint64_t queue_depth = 0;
    uint64_t num_queues = 0;

    // Kernel Launch Parameters
    int block_size = 0;
    int num_blocks = 0;

    // Page Size
    int page_size = 4096; // Default value

    // IO Benchmark Parameters
    IOType io_type = READ;
    IOMethod io_method = SEQUENTIAL;
    int num_io_requests = 0;

    std::string selected_controller_path = "";
};

// Namespace for Input Utilities
namespace InputUtils
{
    // Template function to get user input with prompt
    template <typename T>
    T get_input(const std::string &prompt)
    {
        T value;
        while (true)
        {
            std::cout << prompt;
            std::cin >> value;

            if (std::cin.fail())
            {
                std::cin.clear();                                                   // Clear error flags
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                std::cerr << "Invalid input. Please try again.\n";
            }
            else
            {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard remaining input
                return value;
            }
        }
    }

    // Function to get a selection index from a list of options
    size_t get_selection(const std::string &prompt, const std::vector<std::string> &options)
    {
        size_t selection;
        while (true)
        {
            std::cout << prompt;
            std::cin >> selection;

            if (std::cin.fail() || selection >= options.size())
            {
                std::cin.clear();                                                   // Clear error flags
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                std::cerr << "Invalid selection. Please enter a number between 0 and " << options.size() - 1 << ".\n";
            }
            else
            {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard remaining input
                return selection;
            }
        }
    }
}

// Function to get selection index from a list of options
int get_selection(const std::string &prompt, const std::vector<std::string> &options)
{
    int selection = 0;
    while (true)
    {
        std::cout << prompt;
        std::cin >> selection;

        if (std::cin.fail() || selection < 0 || selection >= options.size())
        {
            std::cin.clear();                                                   // Clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
            std::cerr << "Invalid selection. Please enter a number between 0 and " << options.size() - 1 << ".\n";
        }
        else
        {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard any extra input
            return selection;
        }
    }
}

// Utility Functions

// Function to list available CUDA devices
inline void list_cuda_devices() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Available CUDA Devices:\n";
    for(int i = 0; i < device_count; ++i){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << i << ": " << prop.name 
                  << " | Compute Capability " << prop.major << "." << prop.minor 
                  << " | Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    }
}

// Function to parse command-line arguments, discover/select NVMe controllers, and populate Settings
inline Settings parse_arguments(int argc, char** argv) {
    Settings settings;

    // Supported values
    std::vector<uint64_t> supported_queue_depths = {128, 256, 512, 1024};
    std::vector<uint64_t> supported_num_queues = {1, 2, 4, 8};
    std::vector<std::string> controller_paths = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2"};

    // Parse command-line arguments
    for(int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if(arg == "--device" && i + 1 < argc){
            settings.device = std::stoi(argv[++i]);
        }
        else if(arg == "--controller" && i + 1 < argc){
            settings.controller_index = std::stoul(argv[++i]);
        }
        else if(arg == "--queue-depth" && i + 1 < argc){
            settings.queue_depth = std::stoull(argv[++i]);
        }
        else if(arg == "--num-queues" && i + 1 < argc){
            settings.num_queues = std::stoull(argv[++i]);
        }
        else if(arg == "--block-size" && i + 1 < argc){
            settings.block_size = std::stoi(argv[++i]);
        }
        else if(arg == "--num-blocks" && i + 1 < argc){
            settings.num_blocks = std::stoi(argv[++i]);
        }
        else if(arg == "--page-size" && i + 1 < argc){
            settings.page_size = std::stoi(argv[++i]);
        }
        else if(arg == "--io-type" && i + 1 < argc){
            int temp = std::stoi(argv[++i]);
            if(temp == READ || temp == WRITE){
                settings.io_type = static_cast<IOType>(temp);
            } else {
                std::cerr << "Invalid IO type provided via arguments. Must be 0 (READ) or 1 (WRITE).\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--io-method" && i + 1 < argc){
            int temp = std::stoi(argv[++i]);
            if(temp == SEQUENTIAL || temp == RANDOM){
                settings.io_method = static_cast<IOMethod>(temp);
            } else {
                std::cerr << "Invalid IO method provided via arguments. Must be 0 (SEQUENTIAL) or 1 (RANDOM).\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--num-io" && i + 1 < argc){
            settings.num_io_requests = std::stoi(argv[++i]);
        }
        else if(arg == "--help" || arg == "-h"){
            std::cout << "Usage: ./io_benchmark [options]\n"
                      << "Options:\n"
                      << "  --device <id>           GPU device ID\n"
                      << "  --controller <index>    NVMe controller index\n"
                      << "  --queue-depth <value>   Queue depth (128, 256, 512, 1024)\n"
                      << "  --num-queues <value>    Number of queues (1, 2, 4, 8)\n"
                      << "  --block-size <value>    Threads per block\n"
                      << "  --num-blocks <value>    Number of blocks\n"
                      << "  --page-size <bytes>     Page size (multiple of 4096)\n"
                      << "  --io-type <0|1>         IO type (0: READ, 1: WRITE)\n"
                      << "  --io-method <0|1>       IO method (0: SEQUENTIAL, 1: RANDOM)\n"
                      << "  --num-io <value>        Number of IO requests\n";
            exit(EXIT_SUCCESS);
        }
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // 1. List and Select CUDA Device
    // list_cuda_devices();

    // Get device count
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // If device not set via args, prompt the user
    if(settings.device < 0){
        settings.device = 0; // Default GPU device ID
        if (device_count > 1)
        {
            settings.device = InputUtils::get_input<int>("Enter the device ID of the GPU you want to use: ");

            if (settings.device < 0 || settings.device >= device_count)
            {
                std::cerr << "Invalid device ID entered. Exiting." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    else {
        if (settings.device < 0 || settings.device >= device_count)
        {
            std::cerr << "Invalid device ID provided via arguments. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // 2. Set CUDA Device
    err = cudaSetDevice(settings.device);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to set CUDA device " << settings.device << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // 3. Display Selected GPU Information
    cudaDeviceProp selected_prop;
    err = cudaGetDeviceProperties(&selected_prop, settings.device);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get properties for device " << settings.device << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    char selected_pci_bus_id[16];
    err = cudaDeviceGetPCIBusId(selected_pci_bus_id, sizeof(selected_pci_bus_id), settings.device);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get PCIe Bus ID for device " << settings.device << ": "
                  << cudaGetErrorString(err) << std::endl;
        strcpy(selected_pci_bus_id, "Unknown");
    }

    std::cout << "\nUsing GPU Device " << settings.device << ": " << selected_prop.name
              << " | PCIe Bus ID: " << selected_pci_bus_id << std::endl;

    // 4. Discover and Select NVMe Controllers
    std::vector<std::string> accessible_paths;

    for (const auto &path : controller_paths)
    {
        if (access(path.c_str(), F_OK) != -1)
        {
            accessible_paths.emplace_back(path);
        }
    }

    if (accessible_paths.empty())
    {
        std::cerr << "No accessible NVMe controllers found." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "\nAvailable NVMe Controllers:" << std::endl;
    for (size_t i = 0; i < accessible_paths.size(); ++i)
    {
        std::cout << i << ": " << accessible_paths[i] << std::endl;
    }

    // If controller index not set via args, prompt the user
    if(accessible_paths.size() > 1 && settings.controller_index >= accessible_paths.size()){
        settings.controller_index = InputUtils::get_selection("Enter the index of the controller you want to use: ", accessible_paths);
    }
    else if(accessible_paths.size() == 1){
        settings.controller_index = 0;
        std::cout << "\nUsing NVMe Controller: " << accessible_paths[settings.controller_index] << std::endl;
    }

    // Assign selected controller path
    settings.selected_controller_path = accessible_paths[settings.controller_index];

    // 5. Select Queue Depth and Number of Queues
    // Queue Depth
    if(settings.queue_depth == 0){
        // Display Supported Queue Depths
        std::vector<std::string> qd_options;
        for(auto qd : supported_queue_depths){
            qd_options.emplace_back(std::to_string(qd));
        }
        std::cout << "\nSupported Queue Depths:" << std::endl;
        for (size_t i = 0; i < supported_queue_depths.size(); ++i)
        {
            std::cout << i << ": " << supported_queue_depths[i] << std::endl;
        }

        // Prompt User to Select Queue Depth
        size_t qd_idx = 0;
        qd_idx = InputUtils::get_selection("Select the queue depth by entering the corresponding index: ", qd_options);

        settings.queue_depth = supported_queue_depths[qd_idx];
    }
    else {
        // Validate provided queue_depth
        // if(std::find(supported_queue_depths.begin(), supported_queue_depths.end(), settings.queue_depth) == supported_queue_depths.end()){
        //     std::cerr << "Invalid queue depth provided via arguments. Exiting." << std::endl;
        //     exit(EXIT_FAILURE);
        // }

        // queue-depth is positive integer
        if (settings.queue_depth <= 1)
        {
            std::cerr << "Invalid queue depth entered. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Number of Queues
    if(settings.num_queues == 0){
        // Display Supported Number of Queues
        std::vector<std::string> nq_options;
        for(auto nq : supported_num_queues){
            nq_options.emplace_back(std::to_string(nq));
        }
        std::cout << "\nSupported Number of Queues:" << std::endl;
        for (size_t i = 0; i < supported_num_queues.size(); ++i)
        {
            std::cout << i << ": " << supported_num_queues[i] << std::endl;
        }

        // Prompt User to Select Number of Queues
        size_t nq_idx = 0;
        nq_idx = InputUtils::get_selection("Select the number of queues by entering the corresponding index: ", nq_options);

        settings.num_queues = supported_num_queues[nq_idx];
    }
    else {
        // Validate provided num_queues
        // if less than 1 or not a power of 2
        if (settings.num_queues <= 0 || (settings.num_queues & (settings.num_queues - 1)) != 0)
        {
            std::cerr << "Number of queues must be a positive power of 2. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // 6. Select Kernel Launch Parameters
    if(settings.block_size <= 0){
        settings.block_size = InputUtils::get_input<int>("Enter the number of threads per block: ");
        if (settings.block_size <= 0)
        {
            std::cerr << "Invalid thread count entered. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if(settings.num_blocks <= 0){
        settings.num_blocks = InputUtils::get_input<int>("Enter the number of blocks: ");
        if (settings.num_blocks <= 0)
        {
            std::cerr << "Invalid block count entered. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // 7. Set Page Size
    if(settings.page_size <= 0 || settings.page_size % 4096 != 0){
        while (true)
        {
            std::string page_size_str = InputUtils::get_input<std::string>("Enter the page size (multiple of 4096): ");
            try
            {
                settings.page_size = std::stoi(page_size_str);
                if (settings.page_size > 0 && settings.page_size % 4096 == 0)
                {
                    break;
                }
                else
                {
                    std::cerr << "Page size must be a positive multiple of 4096. Please try again.\n";
                }
            }
            catch (const std::invalid_argument &)
            {
                std::cerr << "Invalid page size entered. Please enter a numeric value.\n";
            }
        }
    }

    // 8. Select IO Benchmark Parameters
    if(settings.io_type != READ && settings.io_type != WRITE){
        settings.io_type = static_cast<IOType>(InputUtils::get_input<int>("Enter the IO type (0: READ, 1: WRITE): "));
        while (settings.io_type != READ && settings.io_type != WRITE)
        {
            std::cerr << "Invalid IO type entered. Please enter 0 for READ or 1 for WRITE.\n";
            settings.io_type = static_cast<IOType>(InputUtils::get_input<int>("Enter the IO type (0: READ, 1: WRITE): "));
        }
    }

    if(settings.io_method != SEQUENTIAL && settings.io_method != RANDOM){
        settings.io_method = static_cast<IOMethod>(InputUtils::get_input<int>("Enter the IO method (0: SEQUENTIAL, 1: RANDOM): "));
        while (settings.io_method != SEQUENTIAL && settings.io_method != RANDOM)
        {
            std::cerr << "Invalid IO method entered. Please enter 0 for SEQUENTIAL or 1 for RANDOM.\n";
            settings.io_method = static_cast<IOMethod>(InputUtils::get_input<int>("Enter the IO method (0: SEQUENTIAL, 1: RANDOM): "));
        }
    }

    if(settings.num_io_requests <= 0){
        settings.num_io_requests = InputUtils::get_input<int>("Enter the number of IO requests: ");
        while (settings.num_io_requests <= 0)
        {
            std::cerr << "Number of IO requests must be positive. Please try again.\n";
            settings.num_io_requests = InputUtils::get_input<int>("Enter the number of IO requests: ");
        }
    }

    return settings;
}

// Function to print all configuration parameters
inline void print_settings(const Settings& settings) {
    std::cout << "\n===== Configuration Parameters =====" << std::endl;
    std::cout << "GPU Device ID     : " << settings.device << std::endl;
    std::cout << "GPU Name          : ";
    
    // Retrieve GPU name
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, settings.device);
    if (err == cudaSuccess) {
        std::cout << prop.name << std::endl;
    }
    else {
        std::cout << "Unknown" << std::endl;
    }

    std::cout << "GPU PCIe Bus ID   : ";
    char pci_bus_id[16];
    err = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), settings.device);
    if (err == cudaSuccess) {
        std::cout << pci_bus_id << std::endl;
    }
    else {
        std::cout << "Unknown" << std::endl;
    }

    std::cout << "NVMe Controller   : " << settings.selected_controller_path << std::endl;
    std::cout << "Queue Depth       : " << settings.queue_depth << std::endl;
    std::cout << "Number of Queues  : " << settings.num_queues << std::endl;
    std::cout << "Threads per Block : " << settings.block_size << std::endl;
    std::cout << "Number of Blocks  : " << settings.num_blocks << std::endl;
    std::cout << "Page Size         : " << settings.page_size << " bytes" << std::endl;
    std::cout << "IO Type           : " << (settings.io_type == READ ? "READ" : "WRITE") << std::endl;
    std::cout << "IO Method         : " << (settings.io_method == SEQUENTIAL ? "SEQUENTIAL" : "RANDOM") << std::endl;
    std::cout << "Number of IO Req. : " << settings.num_io_requests << std::endl;
    std::cout << "====================================\n" << std::endl;
}