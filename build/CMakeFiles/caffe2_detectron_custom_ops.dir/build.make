# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/icubic/detectron

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/icubic/detectron/build

# Include any dependencies generated for this target.
include CMakeFiles/caffe2_detectron_custom_ops.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/caffe2_detectron_custom_ops.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/caffe2_detectron_custom_ops.dir/flags.make

CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o: CMakeFiles/caffe2_detectron_custom_ops.dir/flags.make
CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o: ../detectron/ops/zero_even_op.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/icubic/detectron/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o -c /home/icubic/detectron/detectron/ops/zero_even_op.cc

CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/icubic/detectron/detectron/ops/zero_even_op.cc > CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.i

CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/icubic/detectron/detectron/ops/zero_even_op.cc -o CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.s

CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.requires:

.PHONY : CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.requires

CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.provides: CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.requires
	$(MAKE) -f CMakeFiles/caffe2_detectron_custom_ops.dir/build.make CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.provides.build
.PHONY : CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.provides

CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.provides.build: CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o


# Object files for target caffe2_detectron_custom_ops
caffe2_detectron_custom_ops_OBJECTS = \
"CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o"

# External object files for target caffe2_detectron_custom_ops
caffe2_detectron_custom_ops_EXTERNAL_OBJECTS =

libcaffe2_detectron_custom_ops.so: CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o
libcaffe2_detectron_custom_ops.so: CMakeFiles/caffe2_detectron_custom_ops.dir/build.make
libcaffe2_detectron_custom_ops.so: /home/icubic/pytorch/torch/lib/libc10.so
libcaffe2_detectron_custom_ops.so: CMakeFiles/caffe2_detectron_custom_ops.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/icubic/detectron/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcaffe2_detectron_custom_ops.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffe2_detectron_custom_ops.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/caffe2_detectron_custom_ops.dir/build: libcaffe2_detectron_custom_ops.so

.PHONY : CMakeFiles/caffe2_detectron_custom_ops.dir/build

CMakeFiles/caffe2_detectron_custom_ops.dir/requires: CMakeFiles/caffe2_detectron_custom_ops.dir/detectron/ops/zero_even_op.cc.o.requires

.PHONY : CMakeFiles/caffe2_detectron_custom_ops.dir/requires

CMakeFiles/caffe2_detectron_custom_ops.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/caffe2_detectron_custom_ops.dir/cmake_clean.cmake
.PHONY : CMakeFiles/caffe2_detectron_custom_ops.dir/clean

CMakeFiles/caffe2_detectron_custom_ops.dir/depend:
	cd /home/icubic/detectron/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/icubic/detectron /home/icubic/detectron /home/icubic/detectron/build /home/icubic/detectron/build /home/icubic/detectron/build/CMakeFiles/caffe2_detectron_custom_ops.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/caffe2_detectron_custom_ops.dir/depend

