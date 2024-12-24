cmake_minimum_required(VERSION 3.18)
project(bam LANGUAGES CXX CUDA)

# Set the global runtime output directory to build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set target architecture to 80
set(CMAKE_CUDA_ARCHITECTURES 80)

# Add bam subdirectory
add_subdirectory(external/bam)

# Define an alias for bam's library (optional)
add_library(bam::bam ALIAS libnvm)

# Include directories (shared across all executables)
set(BAM_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/bam/include
    ${CUDAToolkit_INCLUDE_DIRS}  # Ensures CUDA headers are found
    ${CMAKE_CURRENT_SOURCE_DIR}/external/bam/include/freestanding/include
)

# Get all source files in the src/ directory
file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

# Loop over each source file and create an executable target
foreach(SRC_FILE ${SRC_FILES})
    # Extract the base name of the source file (e.g., block-test from block-test.cu)
    get_filename_component(EXEC_NAME ${SRC_FILE} NAME_WE)
    
    # Define an executable with the same name as the base file name
    add_executable(${EXEC_NAME} ${SRC_FILE})

    # Include directories
    target_include_directories(${EXEC_NAME}
        PRIVATE
            ${BAM_INCLUDE_DIRS}
    )

    # Link against bam library and CUDA runtime
    target_link_libraries(${EXEC_NAME}
        PRIVATE
            bam::bam  # Use 'nvm' directly if not using the alias
            CUDA::cudart
    )

    # Set CUDA properties (if necessary)
    set_target_properties(${EXEC_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()
