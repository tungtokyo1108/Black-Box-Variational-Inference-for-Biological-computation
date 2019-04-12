cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Hash_test)

include_directories(include)

set (SOURCES src/Hash.cpp)

add_executable(testHash ${SOURCES})
