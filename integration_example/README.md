# Parth Integration Example

This directory contains a minimal example of how to integrate and use the Parth library in your own project using CMake.

## Overview

Parth is a C++ library for fill-reducing orderings in sparse Cholesky factorization with dynamic sparsity patterns. This integration example demonstrates:

1. How to download and build Parth using CMake's FetchContent
2. How to link your application with Parth
3. How to use Parth's API for computational reuse

## Prerequisites

- **CMake 3.14+**
- **C++17 compatible compiler**
- **Internet connection** (for downloading dependencies)

### Installing CMake on macOS

If CMake is not installed, you can install it using:

```bash
# Using Homebrew
brew install cmake

# Or download from https://cmake.org/download/
```

## Building and Running

### Quick Start

```bash
# Make sure you're in the integration_example directory
cd integration_example

# Build the project
./scripts/build_with_parth.sh

# Run the example
cd build
./use_parth
```

### Manual Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --parallel

# Run
./use_parth
```

## Project Structure

```
integration_example/
├── CMakeLists.txt                 # Main CMake configuration
├── use_parth.cpp                  # Example application using Parth
├── scripts/
│   └── build_with_parth.sh       # Build script
├── cmake/
│   └── recipes/
│       └── parth.cmake           # Parth download/build recipe
└── README.md                     # This file
```

## Key Files

### `cmake/recipes/parth.cmake`

This file contains the CMake recipe for downloading and building Parth from GitHub. It:
- Uses `FetchContent` to download Parth from the main branch
- Disables unnecessary examples and tests for faster builds
- Makes the `Parth::parth` target available

### `CMakeLists.txt`

Minimal CMake configuration that:
- Downloads and builds Parth using the recipe
- Downloads required dependencies (Eigen, CLI11)
- Creates the `use_parth` executable
- Links all dependencies properly

### `use_parth.cpp`

Example application demonstrating:
- Loading sparse matrices
- Computing fill-reducing permutations
- Computational reuse when matrix patterns change slightly

## Integration in Your Project

To use Parth in your own project:

1. **Copy the cmake recipe**: Copy `cmake/recipes/parth.cmake` to your project
2. **Update your CMakeLists.txt**:
   ```cmake
   # Add the recipes directory to module path
   list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes)
   
   # Include Parth
   include(parth)
   
   # Link with your target
   target_link_libraries(your_target PRIVATE Parth::parth)
   ```

3. **Use Parth in your code**:
   ```cpp
   #include <parth/parth.h>
   
   PARTH::ParthAPI parth;
   parth.setMatrix(n, outerPtr, innerPtr, 1);
   std::vector<int> perm;
   parth.computePermutation(perm);
   ```

## Dependencies

This integration example automatically downloads and builds:

- **Parth** - The main library
- **Eigen3** - For matrix operations
- **CLI11** - For command-line parsing
- **METIS** - Required by Parth (downloaded automatically)
- **SuiteSparse** - Required by Parth (system installation needed)

## Troubleshooting

### CMake not found
```bash
brew install cmake
# or download from https://cmake.org/download/
```

### SuiteSparse missing
```bash
# macOS
brew install suite-sparse

# Ubuntu/Debian
sudo apt-get install libsuitesparse-dev

# RHEL/CentOS
sudo yum install suitesparse-devel
```

### Build fails
1. Check that you have a C++17 compatible compiler
2. Ensure internet connection for downloading dependencies
3. Try cleaning the build directory: `rm -rf build`

## Performance Notes

The example demonstrates Parth's computational reuse capability:
- First matrix computation: ~X seconds (cold start)
- Second matrix computation: ~Y seconds (with reuse)

The speedup depends on how similar the sparsity patterns are between matrices.
