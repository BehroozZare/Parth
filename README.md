# Parth: Fill-Reducing Orderings for Sparse Cholesky Factorization

Parth is a C++ library that provides fill-reducing orderings for sparse Cholesky factorizations. It can be used with state-of-the-art solvers such as MKL, Accelerate, and CHOLMOD to improve the efficiency of sparse matrix factorizations by minimizing fill-in during the decomposition process.

## Key Features

- **Fill-reducing orderings**: Minimize fill-in during Cholesky factorization
- **Dynamic mesh support**: Efficiently handle changing mesh topologies with factor reuse
- **Multiple backends**: Works with METIS, AMD, and other ordering algorithms
- **Modern C++ API**: Clean, easy-to-use interface
- **CMake integration**: Easy integration into existing projects
- **Cross-platform**: Supports Linux, macOS, and Windows

## Quick Start

### Installation

#### Option 1: Using CMake FetchContent (Recommended)

Add this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
    parth
    GIT_REPOSITORY https://github.com/BehroozZare/parth.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(parth)

# Link with your target
target_link_libraries(your_target PRIVATE Parth::parth)
```

#### Option 2: Manual Build and Install

```bash
git clone https://github.com/BehroozZare/parth.git
cd parth
mkdir build && cd build
cmake ..
make -j
sudo make install
```

Then in your project:
```cmake
find_package(parth REQUIRED)
target_link_libraries(your_target PRIVATE Parth::parth)
```

### Basic Usage

```cpp
#include <parth/parth.h>

// Create Parth instance
PARTH::Parth parth;
parth.setReorderingType(PARTH::ReorderingType::METIS);

// Set your mesh connectivity (CSR format)
parth.setMeshPointers(n, Mp, Mi);

// Compute permutation
std::vector<int> perm;
parth.computePermutation(perm, 3); // 3 DOFs per node for 3D problems

// Use permutation with your favorite sparse solver...
```

### API Examples

Check out the [examples/api_demos/](examples/api_demos/) directory for complete examples:

- **basic_permutation_demo.cpp**: Shows basic permutation computation
- **dynamic_mesh_demo.cpp**: Demonstrates factor reuse with changing meshes
- **external_project_example/**: Complete example of using Parth in an external project

## API Reference

### Core Class: `PARTH::Parth`

#### Configuration
- `setReorderingType(ReorderingType)`: Set ordering algorithm (METIS, AMD, AUTO)
- `setVerbose(bool)`: Enable/disable verbose output
- `setNDLevels(int)`: Set nested dissection levels
- `setNumberOfCores(int)`: Set number of cores for parallel processing

#### Mesh Input
- `setMeshPointers(n, Mp, Mi)`: Set mesh connectivity in CSR format
- `setMeshPointers(n, Mp, Mi, map)`: Set mesh with DOF mapping for dynamic meshes
- `setNewToOldDOFMap(map)`: Set mapping for factor reuse

#### Permutation Computation
- `computePermutation(perm, dim)`: Compute fill-reducing permutation
- `mapMeshPermToMatrixPerm(mesh_perm, matrix_perm, dim)`: Map to matrix DOFs

#### Analysis
- `getReuse()`: Get factor reuse percentage
- `getNumChanges()`: Get number of topology changes
- `printTiming()`: Print detailed timing information

---

## Dependencies

### Required
- **CMake** 3.14 or newer
- **C++17** compatible compiler
- **Eigen3** (automatically downloaded if not found)

### Optional
- **METIS** (for METIS ordering, automatically detected)
- **Intel MKL** (for high-performance solvers)
- **CHOLMOD** (for CHOLMOD solver backend)
- **OpenMP** (for parallel processing)

### Build Options

Configure the build with these CMake options:

```bash
cmake -DPARTH_SOLVER_WITH_METIS=ON \
      -DPARTH_SOLVER_WITH_MKL=ON \
      -DPARTH_SOLVER_WITH_CHOLMOD=ON \
      -DPARTH_SOLVER_WITH_DEMO=ON \
      ..
```

## Performance Tips

1. **Enable METIS**: Provides the best ordering quality for most problems
2. **Set appropriate ND levels**: Typically 4-8 levels work well
3. **Use factor reuse**: For dynamic meshes, provide DOF mapping to reuse previous factorizations
4. **Match problem dimension**: Set the correct DOF count per node (1 for scalar, 3 for 3D, etc.)

## Research and Benchmarks

### IPC Benchmark

To generate the benchmark used in the Parth paper, please refer to the [IPC benchmark generator repository](https://github.com/BehroozZare/parth-ipc-benchmark-generator.git) for matrix generation. It comes with detailed instructions and a Docker setup for easy matrix generation.

### Remeshing Benchmark

To generate the remeshing benchmark, first download the set of meshes from the [Oded Stein Meshes repository](https://github.com/odedstein/meshes). Also, please make sure to cite their repository if you use their meshes—they saved my life! You can then feed these meshes to the `PatchRemeshDemo.cpp` code. (I will provide detailed documentation and clean code soon. For now, I’m just drowning in tasks!)

**Notes for this version of the code:**

You can use the Docker setup provided in the IPC benchmark repository to build this code—the dependencies are the same.

CHOLMOD usage on macOS has some issues in this codebase because I modified the CHOLMOD source for some experiments, which led to some library linking problems. Please use Accelerate on macOS. On Linux, everything should work fine.

Honestly though, the code is almost well-structured! So you can still use it in its current state (hopefully 🙂).

---

## Citation

If you use Parth or build upon this work, please cite:

Behrooz Zarebavani, Danny M. Kaufman, David I.W. Levin, and Maryam Mehri Dehnavi. 2025. *Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns*. ACM Trans. Graph. 44, 4 (August 2025), 17 pages. [https://arxiv.org/pdf/2501.04011](https://arxiv.org/pdf/2501.04011)
