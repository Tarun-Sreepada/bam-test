cmake_minimum_required(VERSION 3.18)
project(bam LANGUAGES CXX CUDA)

# Set the build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set the global runtime output directory to build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ and CUDA optimization flags to -O3 for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Alternatively, to force -O3 regardless of build type:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set target architecture to 80 (e.g., NVIDIA Ampere GPUs)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Add bam subdirectory (assumes external/bam has its own CMakeLists.txt)
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

# Function to create executables with and without CLOCK flag
function(add_bam_executables src_file)
    # Extract the base name of the source file (e.g., block-test from block-test.cu)
    get_filename_component(EXEC_NAME ${src_file} NAME_WE)

    # Define suffixes for the two variants
    set(SUFFIXES "noclock" "clock")

    foreach(SUFFIX IN LISTS SUFFIXES)
        # Construct the executable name by appending the suffix
        set(TARGET_NAME "${EXEC_NAME}_${SUFFIX}")

        # Define the executable
        add_executable(${TARGET_NAME} ${src_file})

        # Include directories
        target_include_directories(${TARGET_NAME}
            PRIVATE
                ${BAM_INCLUDE_DIRS}
        )

        # Link against bam library and CUDA runtime
        target_link_libraries(${TARGET_NAME}
            PRIVATE
                bam::bam  # Use 'bam::bam' alias
                CUDA::cudart
        )

        # Set CUDA properties (if necessary)
        set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        # Apply -O3 optimization flag (already set globally, but redundant here for clarity)
        target_compile_options(${TARGET_NAME} PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-O3> 
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )

        # Conditionally add the CLOCK compile definition
        if(SUFFIX STREQUAL "clock")
            target_compile_definitions(${TARGET_NAME} PRIVATE CLOCK)
            message(STATUS "Configured ${TARGET_NAME} with CLOCK flag.")
        else()
            message(STATUS "Configured ${TARGET_NAME} without CLOCK flag.")
        endif()

    endforeach()
endfunction()

# Loop over each source file and create executables
foreach(SRC_FILE ${SRC_FILES})
    add_bam_executables(${SRC_FILE})
endforeach()
