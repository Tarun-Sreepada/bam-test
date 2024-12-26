# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /export/home1/ltarun/bam-test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /export/home1/ltarun/bam-test/build

# Include any dependencies generated for this target.
include external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/compiler_depend.make

# Include the progress variables for this target.
include external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/progress.make

# Include the compile flags for this target's objects.
include external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/flags.make

external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o: /export/home1/ltarun/bam-test/external/bam/benchmarks/cache/main.cu
external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o: external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o.depend
external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o: external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o.Release.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o"
	cd /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir && /usr/local/bin/cmake -E make_directory /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir//.
	cd /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir//./cache-benchmark-module_generated_main.cu.o -D generated_cubin_file:STRING=/export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir//./cache-benchmark-module_generated_main.cu.o.cubin.txt -P /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir//cache-benchmark-module_generated_main.cu.o.Release.cmake

# Object files for target cache-benchmark-module
cache__benchmark__module_OBJECTS =

# External object files for target cache-benchmark-module
cache__benchmark__module_EXTERNAL_OBJECTS = \
"/export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o"

bin/nvm-cache-bench: external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o
bin/nvm-cache-bench: external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/build.make
bin/nvm-cache-bench: /usr/local/cuda-12.6/lib64/libcudart_static.a
bin/nvm-cache-bench: /usr/lib/x86_64-linux-gnu/librt.a
bin/nvm-cache-bench: lib/libnvm.so
bin/nvm-cache-bench: /usr/local/cuda-12.6/lib64/libcudart_static.a
bin/nvm-cache-bench: /usr/lib/x86_64-linux-gnu/librt.a
bin/nvm-cache-bench: /usr/local/cuda-12.6/lib64/libcudart_static.a
bin/nvm-cache-bench: /usr/lib/x86_64-linux-gnu/librt.a
bin/nvm-cache-bench: external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../bin/nvm-cache-bench"
	cd /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cache-benchmark-module.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/build: bin/nvm-cache-bench
.PHONY : external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/build

external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/clean:
	cd /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache && $(CMAKE_COMMAND) -P CMakeFiles/cache-benchmark-module.dir/cmake_clean.cmake
.PHONY : external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/clean

external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/depend: external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/cache-benchmark-module_generated_main.cu.o
	cd /export/home1/ltarun/bam-test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/home1/ltarun/bam-test /export/home1/ltarun/bam-test/external/bam/benchmarks/cache /export/home1/ltarun/bam-test/build /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache /export/home1/ltarun/bam-test/build/external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/bam/benchmarks/cache/CMakeFiles/cache-benchmark-module.dir/depend

