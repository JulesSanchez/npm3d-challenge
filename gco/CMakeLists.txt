cmake_minimum_required(VERSION 3.10)

project(gco)  # set project name

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Main main.cpp)
target_sources(Main PUBLIC src)
target_include_directories(Main PUBLIC src)
