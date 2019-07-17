cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

#project(Random_Matrix)
set(Random_Matrix ${CMAKE_CURRENT_LIST_DIR})

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "-03 -Wall -Wextra")

include_directories("Eigen3.3.7")
include_directories(include)
include_directories(${PROJECT_SOURCE_DIR})

#file(GLOB_RECURSE Random_Matrix_SOURCES "src/*.cpp")
#file(GLOB_RECURSE Random_Matrix_HEADERS "include/*.h")

#set (Random_Matrix_INCLUDE_DIRS "")
#foreach(_headerFile ${Random_Matrix_HEADERS})
#    get_filename_component(_dir ${_headerFile} PATH) 
#    list (APPEND Random_Matrix_INCLUDE_DIRS ${_dir})   
#endforeach()
#list(REMOVE_DUPLICATES Random_Matrix_INCLUDE_DIRS)

#add_executable(testMatrix ${Random_Matrix_SOURCES})
#target_include_directories(testMatrix PRIVATE ${Random_Matrix_INCLUDE_DIRS})

set (src src/SparseRandomMatrix.cpp
         src/utils.cpp
         src/main.cpp)
         
set (include include/SparseRandomMatrix.h
             include/utils.h)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

add_executable(testMatrix ${src} ${include})
target_include_directories(testMatrix PRIVATE include ${Random_Matrix})
