# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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
CMAKE_COMMAND = /root/miniconda3/envs/lane-det/bin/cmake

# The command to remove a file.
RM = /root/miniconda3/envs/lane-det/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/work/Ultra-Fast-Lane-Detection-master/opencv-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/work/Ultra-Fast-Lane-Detection-master/build

# Include any dependencies generated for this target.
include modules/flann/CMakeFiles/opencv_flann.dir/depend.make

# Include the progress variables for this target.
include modules/flann/CMakeFiles/opencv_flann.dir/progress.make

# Include the compile flags for this target's objects.
include modules/flann/CMakeFiles/opencv_flann.dir/flags.make

modules/flann/CMakeFiles/opencv_flann.dir/src/flann.cpp.o: modules/flann/CMakeFiles/opencv_flann.dir/flags.make
modules/flann/CMakeFiles/opencv_flann.dir/src/flann.cpp.o: /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/flann.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/work/Ultra-Fast-Lane-Detection-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/flann/CMakeFiles/opencv_flann.dir/src/flann.cpp.o"
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && /usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_flann.dir/src/flann.cpp.o -c /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/flann.cpp

modules/flann/CMakeFiles/opencv_flann.dir/src/flann.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_flann.dir/src/flann.cpp.i"
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && /usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/flann.cpp > CMakeFiles/opencv_flann.dir/src/flann.cpp.i

modules/flann/CMakeFiles/opencv_flann.dir/src/flann.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_flann.dir/src/flann.cpp.s"
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && /usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/flann.cpp -o CMakeFiles/opencv_flann.dir/src/flann.cpp.s

modules/flann/CMakeFiles/opencv_flann.dir/src/miniflann.cpp.o: modules/flann/CMakeFiles/opencv_flann.dir/flags.make
modules/flann/CMakeFiles/opencv_flann.dir/src/miniflann.cpp.o: /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/miniflann.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/work/Ultra-Fast-Lane-Detection-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/flann/CMakeFiles/opencv_flann.dir/src/miniflann.cpp.o"
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && /usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_flann.dir/src/miniflann.cpp.o -c /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/miniflann.cpp

modules/flann/CMakeFiles/opencv_flann.dir/src/miniflann.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_flann.dir/src/miniflann.cpp.i"
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && /usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/miniflann.cpp > CMakeFiles/opencv_flann.dir/src/miniflann.cpp.i

modules/flann/CMakeFiles/opencv_flann.dir/src/miniflann.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_flann.dir/src/miniflann.cpp.s"
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && /usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann/src/miniflann.cpp -o CMakeFiles/opencv_flann.dir/src/miniflann.cpp.s

# Object files for target opencv_flann
opencv_flann_OBJECTS = \
"CMakeFiles/opencv_flann.dir/src/flann.cpp.o" \
"CMakeFiles/opencv_flann.dir/src/miniflann.cpp.o"

# External object files for target opencv_flann
opencv_flann_EXTERNAL_OBJECTS =

lib/libopencv_flann.so.4.5.2: modules/flann/CMakeFiles/opencv_flann.dir/src/flann.cpp.o
lib/libopencv_flann.so.4.5.2: modules/flann/CMakeFiles/opencv_flann.dir/src/miniflann.cpp.o
lib/libopencv_flann.so.4.5.2: modules/flann/CMakeFiles/opencv_flann.dir/build.make
lib/libopencv_flann.so.4.5.2: lib/libopencv_core.so.4.5.2
lib/libopencv_flann.so.4.5.2: 3rdparty/lib/libippiw.a
lib/libopencv_flann.so.4.5.2: 3rdparty/ippicv/ippicv_lnx/icv/lib/intel64/libippicv.a
lib/libopencv_flann.so.4.5.2: modules/flann/CMakeFiles/opencv_flann.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/work/Ultra-Fast-Lane-Detection-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../../lib/libopencv_flann.so"
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_flann.dir/link.txt --verbose=$(VERBOSE)
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && $(CMAKE_COMMAND) -E cmake_symlink_library ../../lib/libopencv_flann.so.4.5.2 ../../lib/libopencv_flann.so.4.5 ../../lib/libopencv_flann.so

lib/libopencv_flann.so.4.5: lib/libopencv_flann.so.4.5.2
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libopencv_flann.so.4.5

lib/libopencv_flann.so: lib/libopencv_flann.so.4.5.2
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libopencv_flann.so

# Rule to build all files generated by this target.
modules/flann/CMakeFiles/opencv_flann.dir/build: lib/libopencv_flann.so

.PHONY : modules/flann/CMakeFiles/opencv_flann.dir/build

modules/flann/CMakeFiles/opencv_flann.dir/clean:
	cd /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann && $(CMAKE_COMMAND) -P CMakeFiles/opencv_flann.dir/cmake_clean.cmake
.PHONY : modules/flann/CMakeFiles/opencv_flann.dir/clean

modules/flann/CMakeFiles/opencv_flann.dir/depend:
	cd /root/work/Ultra-Fast-Lane-Detection-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/work/Ultra-Fast-Lane-Detection-master/opencv-master /root/work/Ultra-Fast-Lane-Detection-master/opencv-master/modules/flann /root/work/Ultra-Fast-Lane-Detection-master/build /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann /root/work/Ultra-Fast-Lane-Detection-master/build/modules/flann/CMakeFiles/opencv_flann.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/flann/CMakeFiles/opencv_flann.dir/depend

