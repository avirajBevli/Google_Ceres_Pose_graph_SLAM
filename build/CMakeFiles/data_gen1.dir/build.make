# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/aviraj/my_ceres

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aviraj/my_ceres/build

# Include any dependencies generated for this target.
include CMakeFiles/data_gen1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/data_gen1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/data_gen1.dir/flags.make

CMakeFiles/data_gen1.dir/data_gen1.cpp.o: CMakeFiles/data_gen1.dir/flags.make
CMakeFiles/data_gen1.dir/data_gen1.cpp.o: ../data_gen1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aviraj/my_ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/data_gen1.dir/data_gen1.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/data_gen1.dir/data_gen1.cpp.o -c /home/aviraj/my_ceres/data_gen1.cpp

CMakeFiles/data_gen1.dir/data_gen1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/data_gen1.dir/data_gen1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aviraj/my_ceres/data_gen1.cpp > CMakeFiles/data_gen1.dir/data_gen1.cpp.i

CMakeFiles/data_gen1.dir/data_gen1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/data_gen1.dir/data_gen1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aviraj/my_ceres/data_gen1.cpp -o CMakeFiles/data_gen1.dir/data_gen1.cpp.s

CMakeFiles/data_gen1.dir/data_gen1.cpp.o.requires:

.PHONY : CMakeFiles/data_gen1.dir/data_gen1.cpp.o.requires

CMakeFiles/data_gen1.dir/data_gen1.cpp.o.provides: CMakeFiles/data_gen1.dir/data_gen1.cpp.o.requires
	$(MAKE) -f CMakeFiles/data_gen1.dir/build.make CMakeFiles/data_gen1.dir/data_gen1.cpp.o.provides.build
.PHONY : CMakeFiles/data_gen1.dir/data_gen1.cpp.o.provides

CMakeFiles/data_gen1.dir/data_gen1.cpp.o.provides.build: CMakeFiles/data_gen1.dir/data_gen1.cpp.o


# Object files for target data_gen1
data_gen1_OBJECTS = \
"CMakeFiles/data_gen1.dir/data_gen1.cpp.o"

# External object files for target data_gen1
data_gen1_EXTERNAL_OBJECTS =

data_gen1: CMakeFiles/data_gen1.dir/data_gen1.cpp.o
data_gen1: CMakeFiles/data_gen1.dir/build.make
data_gen1: /usr/local/lib/libceres.a
data_gen1: /usr/lib/x86_64-linux-gnu/libglog.so
data_gen1: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
data_gen1: /usr/lib/x86_64-linux-gnu/libspqr.so
data_gen1: /usr/lib/x86_64-linux-gnu/libcholmod.so
data_gen1: /usr/lib/x86_64-linux-gnu/libccolamd.so
data_gen1: /usr/lib/x86_64-linux-gnu/libcamd.so
data_gen1: /usr/lib/x86_64-linux-gnu/libcolamd.so
data_gen1: /usr/lib/x86_64-linux-gnu/libamd.so
data_gen1: /usr/lib/x86_64-linux-gnu/liblapack.so
data_gen1: /usr/lib/x86_64-linux-gnu/libf77blas.so
data_gen1: /usr/lib/x86_64-linux-gnu/libatlas.so
data_gen1: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
data_gen1: /usr/lib/x86_64-linux-gnu/librt.so
data_gen1: /usr/lib/x86_64-linux-gnu/libcxsparse.so
data_gen1: /usr/lib/x86_64-linux-gnu/liblapack.so
data_gen1: /usr/lib/x86_64-linux-gnu/libf77blas.so
data_gen1: /usr/lib/x86_64-linux-gnu/libatlas.so
data_gen1: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
data_gen1: /usr/lib/x86_64-linux-gnu/librt.so
data_gen1: /usr/lib/x86_64-linux-gnu/libcxsparse.so
data_gen1: CMakeFiles/data_gen1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aviraj/my_ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable data_gen1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/data_gen1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/data_gen1.dir/build: data_gen1

.PHONY : CMakeFiles/data_gen1.dir/build

CMakeFiles/data_gen1.dir/requires: CMakeFiles/data_gen1.dir/data_gen1.cpp.o.requires

.PHONY : CMakeFiles/data_gen1.dir/requires

CMakeFiles/data_gen1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/data_gen1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/data_gen1.dir/clean

CMakeFiles/data_gen1.dir/depend:
	cd /home/aviraj/my_ceres/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aviraj/my_ceres /home/aviraj/my_ceres /home/aviraj/my_ceres/build /home/aviraj/my_ceres/build /home/aviraj/my_ceres/build/CMakeFiles/data_gen1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/data_gen1.dir/depend
