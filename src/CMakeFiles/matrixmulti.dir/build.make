# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/beren/repositorios/mac0431-ep4/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/beren/repositorios/mac0431-ep4/src

# Include any dependencies generated for this target.
include CMakeFiles/matrixmulti.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matrixmulti.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrixmulti.dir/flags.make

CMakeFiles/matrixmulti.dir/opcl.c.o: CMakeFiles/matrixmulti.dir/flags.make
CMakeFiles/matrixmulti.dir/opcl.c.o: opcl.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/beren/repositorios/mac0431-ep4/src/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/matrixmulti.dir/opcl.c.o"
	/usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/matrixmulti.dir/opcl.c.o   -c /home/beren/repositorios/mac0431-ep4/src/opcl.c

CMakeFiles/matrixmulti.dir/opcl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/matrixmulti.dir/opcl.c.i"
	/usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -E /home/beren/repositorios/mac0431-ep4/src/opcl.c > CMakeFiles/matrixmulti.dir/opcl.c.i

CMakeFiles/matrixmulti.dir/opcl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/matrixmulti.dir/opcl.c.s"
	/usr/bin/gcc  $(C_DEFINES) $(C_FLAGS) -S /home/beren/repositorios/mac0431-ep4/src/opcl.c -o CMakeFiles/matrixmulti.dir/opcl.c.s

CMakeFiles/matrixmulti.dir/opcl.c.o.requires:
.PHONY : CMakeFiles/matrixmulti.dir/opcl.c.o.requires

CMakeFiles/matrixmulti.dir/opcl.c.o.provides: CMakeFiles/matrixmulti.dir/opcl.c.o.requires
	$(MAKE) -f CMakeFiles/matrixmulti.dir/build.make CMakeFiles/matrixmulti.dir/opcl.c.o.provides.build
.PHONY : CMakeFiles/matrixmulti.dir/opcl.c.o.provides

CMakeFiles/matrixmulti.dir/opcl.c.o.provides.build: CMakeFiles/matrixmulti.dir/opcl.c.o

# Object files for target matrixmulti
matrixmulti_OBJECTS = \
"CMakeFiles/matrixmulti.dir/opcl.c.o"

# External object files for target matrixmulti
matrixmulti_EXTERNAL_OBJECTS =

libmatrixmulti.a: CMakeFiles/matrixmulti.dir/opcl.c.o
libmatrixmulti.a: CMakeFiles/matrixmulti.dir/build.make
libmatrixmulti.a: CMakeFiles/matrixmulti.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C static library libmatrixmulti.a"
	$(CMAKE_COMMAND) -P CMakeFiles/matrixmulti.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrixmulti.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrixmulti.dir/build: libmatrixmulti.a
.PHONY : CMakeFiles/matrixmulti.dir/build

CMakeFiles/matrixmulti.dir/requires: CMakeFiles/matrixmulti.dir/opcl.c.o.requires
.PHONY : CMakeFiles/matrixmulti.dir/requires

CMakeFiles/matrixmulti.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrixmulti.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrixmulti.dir/clean

CMakeFiles/matrixmulti.dir/depend:
	cd /home/beren/repositorios/mac0431-ep4/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/beren/repositorios/mac0431-ep4/src /home/beren/repositorios/mac0431-ep4/src /home/beren/repositorios/mac0431-ep4/src /home/beren/repositorios/mac0431-ep4/src /home/beren/repositorios/mac0431-ep4/src/CMakeFiles/matrixmulti.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrixmulti.dir/depend

