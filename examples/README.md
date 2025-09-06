# Parth Examples

This directory contains various examples showing how to use the Parth library.

## API Demos (`api_demos/`)

These examples demonstrate the core Parth API:

### `basic_permutation_demo.cpp`
- Shows how to set up a simple mesh
- Computes permutations using different algorithms
- Demonstrates single and multi-DOF problems
- **Usage**: Shows basic permutation computation workflow

### `dynamic_mesh_demo.cpp`
- Demonstrates mesh refinement scenarios
- Shows factor reuse capabilities
- Tracks topology changes and reuse statistics
- **Usage**: Shows how to handle changing mesh topologies efficiently

### Building the API Demos

```bash
cd api_demos
mkdir build && cd build
cmake ..
make

# Run the demos
./basic_permutation_demo
./dynamic_mesh_demo
```

## External Project Example (`external_project_example/`)

Shows how to integrate Parth into your own projects:

- **`CMakeLists.txt`**: Complete example using FetchContent
- **`main.cpp`**: Simple application using Parth

### Building the External Project Example

```bash
cd external_project_example
mkdir build && cd build
cmake ..
make

# Run the example
./my_app
```

This example demonstrates the recommended way to include Parth in your projects using CMake's FetchContent.

## Original Research Demos (`../demo/`)

The `../demo/` directory contains the original research implementations and benchmarking code used in the Parth paper. These are more complex examples that demonstrate the full capabilities of the framework with various solvers and applications.

## Getting Started

1. **New users**: Start with `basic_permutation_demo.cpp`
2. **Dynamic meshes**: Look at `dynamic_mesh_demo.cpp`
3. **Integration**: Use `external_project_example/` as a template

For detailed API documentation, see the main [README.md](../README.md) file.
