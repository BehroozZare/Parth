# Example CMakeLists.txt showing how to use Parth in an external project
# This demonstrates downloading and using Parth via FetchContent

cmake_minimum_required(VERSION 3.14)
project(MyProjectUsingParth VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use FetchContent to download and build Parth
include(FetchContent)

FetchContent_Declare(
    parth
    GIT_REPOSITORY https://github.com/BehroozZare/parth.git  # Replace with actual repo URL
    GIT_TAG        main  # or specific version tag like "v1.0.0"
    # Alternatively, you can use a local path during development:
    # SOURCE_DIR     /path/to/local/parth
)

# Configure Parth options (optional)
set(PARTH_SOLVER_WITH_DEMO OFF CACHE BOOL "Don't build Parth demos")
set(PARTH_SOLVER_WITH_CHOLMOD OFF CACHE BOOL "Disable CHOLMOD for simpler build")
set(PARTH_SOLVER_WITH_MKL OFF CACHE BOOL "Disable MKL for simpler build")
set(PARTH_SOLVER_WITH_OPENGL OFF CACHE BOOL "Disable OpenGL for simpler build")

FetchContent_MakeAvailable(parth)

# Your application
add_executable(my_app main.cpp)

# Link with Parth
target_link_libraries(my_app PRIVATE Parth::parth)

# Print success message
message(STATUS "Successfully configured project with Parth")
message(STATUS "Build your project with: cmake --build . --target my_app")
