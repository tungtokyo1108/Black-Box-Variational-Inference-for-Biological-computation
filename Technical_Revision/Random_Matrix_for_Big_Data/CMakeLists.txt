cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Random_Matrix)

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "-03 -Wall -Wextra")

include_directories("Eigen3.3.7")

include_directories(include)

set (SOURCES src/main.cpp)

add_executable(testMatrix ${SOURCES})

