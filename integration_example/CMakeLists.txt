# Parth Integration Example
# Minimal CMake setup for using Parth library
cmake_minimum_required(VERSION 3.14)

project(parth_integration_example DESCRIPTION "Integration example for Parth library" LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add our cmake module paths for finding recipes
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find)

# Include Parth library
include(parth)

# Include Eigen (required by use_parth.cpp for matrix operations)
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)
add_library(Eigen3::Eigen ALIAS eigen)

# Include CLI11 (required by use_parth.cpp for command line parsing)
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cli11)

# Create the example executable
add_executable(use_parth use_parth.cpp)

# Link libraries
target_link_libraries(use_parth PRIVATE 
    Parth::parth
    Eigen3::Eigen
    CLI11::CLI11
)

# Set include directories
target_include_directories(use_parth PRIVATE 
    ${eigen_SOURCE_DIR}
)

message(STATUS "Parth integration example configured successfully")
message(STATUS "Executable target: use_parth")
message(STATUS "To build: make use_parth")
