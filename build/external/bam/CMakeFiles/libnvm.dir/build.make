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
include external/bam/CMakeFiles/libnvm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/bam/CMakeFiles/libnvm.dir/compiler_depend.make

# Include the progress variables for this target.
include external/bam/CMakeFiles/libnvm.dir/progress.make

# Include the compile flags for this target's objects.
include external/bam/CMakeFiles/libnvm.dir/flags.make

external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/admin.cpp
external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.o -MF CMakeFiles/libnvm.dir/src/admin.cpp.o.d -o CMakeFiles/libnvm.dir/src/admin.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/admin.cpp

external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/admin.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/admin.cpp > CMakeFiles/libnvm.dir/src/admin.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/admin.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/admin.cpp -o CMakeFiles/libnvm.dir/src/admin.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/ctrl.cpp
external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.o -MF CMakeFiles/libnvm.dir/src/ctrl.cpp.o.d -o CMakeFiles/libnvm.dir/src/ctrl.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/ctrl.cpp

external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/ctrl.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/ctrl.cpp > CMakeFiles/libnvm.dir/src/ctrl.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/ctrl.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/ctrl.cpp -o CMakeFiles/libnvm.dir/src/ctrl.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/dma.cpp
external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.o -MF CMakeFiles/libnvm.dir/src/dma.cpp.o.d -o CMakeFiles/libnvm.dir/src/dma.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/dma.cpp

external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/dma.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/dma.cpp > CMakeFiles/libnvm.dir/src/dma.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/dma.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/dma.cpp -o CMakeFiles/libnvm.dir/src/dma.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/error.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/error.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/error.cpp
external/bam/CMakeFiles/libnvm.dir/src/error.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/error.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/error.cpp.o -MF CMakeFiles/libnvm.dir/src/error.cpp.o.d -o CMakeFiles/libnvm.dir/src/error.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/error.cpp

external/bam/CMakeFiles/libnvm.dir/src/error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/error.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/error.cpp > CMakeFiles/libnvm.dir/src/error.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/error.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/error.cpp -o CMakeFiles/libnvm.dir/src/error.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/mutex.cpp
external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.o -MF CMakeFiles/libnvm.dir/src/mutex.cpp.o.d -o CMakeFiles/libnvm.dir/src/mutex.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/mutex.cpp

external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/mutex.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/mutex.cpp > CMakeFiles/libnvm.dir/src/mutex.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/mutex.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/mutex.cpp -o CMakeFiles/libnvm.dir/src/mutex.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/queue.cpp
external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.o -MF CMakeFiles/libnvm.dir/src/queue.cpp.o.d -o CMakeFiles/libnvm.dir/src/queue.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/queue.cpp

external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/queue.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/queue.cpp > CMakeFiles/libnvm.dir/src/queue.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/queue.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/queue.cpp -o CMakeFiles/libnvm.dir/src/queue.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/rpc.cpp
external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.o -MF CMakeFiles/libnvm.dir/src/rpc.cpp.o.d -o CMakeFiles/libnvm.dir/src/rpc.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/rpc.cpp

external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/rpc.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/rpc.cpp > CMakeFiles/libnvm.dir/src/rpc.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/rpc.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/rpc.cpp -o CMakeFiles/libnvm.dir/src/rpc.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/linux/device.cpp
external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.o -MF CMakeFiles/libnvm.dir/src/linux/device.cpp.o.d -o CMakeFiles/libnvm.dir/src/linux/device.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/linux/device.cpp

external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/linux/device.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/linux/device.cpp > CMakeFiles/libnvm.dir/src/linux/device.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/linux/device.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/linux/device.cpp -o CMakeFiles/libnvm.dir/src/linux/device.cpp.s

external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.o: external/bam/CMakeFiles/libnvm.dir/flags.make
external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.o: /export/home1/ltarun/bam-test/external/bam/src/linux/dma.cpp
external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.o: external/bam/CMakeFiles/libnvm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.o"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.o -MF CMakeFiles/libnvm.dir/src/linux/dma.cpp.o.d -o CMakeFiles/libnvm.dir/src/linux/dma.cpp.o -c /export/home1/ltarun/bam-test/external/bam/src/linux/dma.cpp

external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libnvm.dir/src/linux/dma.cpp.i"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /export/home1/ltarun/bam-test/external/bam/src/linux/dma.cpp > CMakeFiles/libnvm.dir/src/linux/dma.cpp.i

external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libnvm.dir/src/linux/dma.cpp.s"
	cd /export/home1/ltarun/bam-test/build/external/bam && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /export/home1/ltarun/bam-test/external/bam/src/linux/dma.cpp -o CMakeFiles/libnvm.dir/src/linux/dma.cpp.s

# Object files for target libnvm
libnvm_OBJECTS = \
"CMakeFiles/libnvm.dir/src/admin.cpp.o" \
"CMakeFiles/libnvm.dir/src/ctrl.cpp.o" \
"CMakeFiles/libnvm.dir/src/dma.cpp.o" \
"CMakeFiles/libnvm.dir/src/error.cpp.o" \
"CMakeFiles/libnvm.dir/src/mutex.cpp.o" \
"CMakeFiles/libnvm.dir/src/queue.cpp.o" \
"CMakeFiles/libnvm.dir/src/rpc.cpp.o" \
"CMakeFiles/libnvm.dir/src/linux/device.cpp.o" \
"CMakeFiles/libnvm.dir/src/linux/dma.cpp.o"

# External object files for target libnvm
libnvm_EXTERNAL_OBJECTS =

lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/admin.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/ctrl.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/dma.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/error.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/mutex.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/queue.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/rpc.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/linux/device.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/src/linux/dma.cpp.o
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/build.make
lib/libnvm.so: /usr/local/cuda-12.6/lib64/libcudart_static.a
lib/libnvm.so: /usr/lib/x86_64-linux-gnu/librt.a
lib/libnvm.so: /usr/local/cuda-12.6/lib64/libcudart_static.a
lib/libnvm.so: /usr/lib/x86_64-linux-gnu/librt.a
lib/libnvm.so: external/bam/CMakeFiles/libnvm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/export/home1/ltarun/bam-test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library ../../lib/libnvm.so"
	cd /export/home1/ltarun/bam-test/build/external/bam && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libnvm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/bam/CMakeFiles/libnvm.dir/build: lib/libnvm.so
.PHONY : external/bam/CMakeFiles/libnvm.dir/build

external/bam/CMakeFiles/libnvm.dir/clean:
	cd /export/home1/ltarun/bam-test/build/external/bam && $(CMAKE_COMMAND) -P CMakeFiles/libnvm.dir/cmake_clean.cmake
.PHONY : external/bam/CMakeFiles/libnvm.dir/clean

external/bam/CMakeFiles/libnvm.dir/depend:
	cd /export/home1/ltarun/bam-test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/home1/ltarun/bam-test /export/home1/ltarun/bam-test/external/bam /export/home1/ltarun/bam-test/build /export/home1/ltarun/bam-test/build/external/bam /export/home1/ltarun/bam-test/build/external/bam/CMakeFiles/libnvm.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/bam/CMakeFiles/libnvm.dir/depend

