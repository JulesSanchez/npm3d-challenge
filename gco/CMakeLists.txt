cmake_minimum_required(VERSION 3.10)

project(gco)  # set project name

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRCDIR src/)

set(SOURCES
    ${SRCDIR}/GCoptimization.cpp
    ${SRCDIR}/graph.cpp
    ${SRCDIR}/LinkedBlockList.cpp
    ${SRCDIR}/maxflow.cpp
    )

include_directories(src)

add_executable(Main main.cpp)
target_sources(Main PUBLIC ${SOURCES})