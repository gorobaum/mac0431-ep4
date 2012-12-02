CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
project(ep4)

find_library(OPENCL OpenCL)
message("OpenCL Library == ${OPENCL}")

include_directories(./../opencl12)
message("include directories: \n")
get_directory_property(includes INCLUDE_DIRECTORIES)
message("${includes}\n")

ADD_DEFINITIONS(-DCL_SOURCE_DIR="${matrixmulti_SOURCE_DIR}")

set(multilib opcl.c)
set(opcllib matrixmulti)
add_library(${opcllib} ${multilib})

set(exec ep4)
add_executable(${exec} main.c)

target_link_libraries(${exec} ${opcllib} ${OPENCL})
