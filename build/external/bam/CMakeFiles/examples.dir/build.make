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

# Utility rule file for examples.

# Include any custom commands dependencies for this target.
include external/bam/CMakeFiles/examples.dir/compiler_depend.make

# Include the progress variables for this target.
include external/bam/CMakeFiles/examples.dir/progress.make

external/bam/CMakeFiles/examples:

examples: external/bam/CMakeFiles/examples
examples: external/bam/CMakeFiles/examples.dir/build.make
.PHONY : examples

# Rule to build all files generated by this target.
external/bam/CMakeFiles/examples.dir/build: examples
.PHONY : external/bam/CMakeFiles/examples.dir/build

external/bam/CMakeFiles/examples.dir/clean:
	cd /export/home1/ltarun/bam-test/build/external/bam && $(CMAKE_COMMAND) -P CMakeFiles/examples.dir/cmake_clean.cmake
.PHONY : external/bam/CMakeFiles/examples.dir/clean

external/bam/CMakeFiles/examples.dir/depend:
	cd /export/home1/ltarun/bam-test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /export/home1/ltarun/bam-test /export/home1/ltarun/bam-test/external/bam /export/home1/ltarun/bam-test/build /export/home1/ltarun/bam-test/build/external/bam /export/home1/ltarun/bam-test/build/external/bam/CMakeFiles/examples.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/bam/CMakeFiles/examples.dir/depend

