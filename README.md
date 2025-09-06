# Parth: Fill-Reducing Orderings for Sparse Cholesky Factorization

Parth is a C++ library that provides fill-reducing orderings for sparse Cholesky factorizations when sparsity pattern is dynamic. It can be used with state-of-the-art solvers such as MKL, Accelerate, and CHOLMOD to improve the efficiency of sparse matrix factorizations by minimizing fill-in during the decomposition process.

## Key Features

- **Fill-reducing orderings**: Minimize fill-in during Cholesky factorization
- **Dynamic mesh support**: Efficiently handle changing mesh topologies
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
PARTH::ParthAPI parth;
parth.setReorderingType(PARTH::METIS);

// Set your mesh connectivity (CSR format)
parth.setMesh(n, Mp, Mi);

// Compute permutation
std::vector<int> perm;
parth.computePermutation(perm, 3); // 3 DOFs per node for 3D problems

// Use permutation with your favorite sparse solver...
```

### API Examples

Check out the [examples/api_demos/](examples/api_demos/) directory for complete examples:

- **quick_start.cpp**: Shows basic permutation computation
- **intermediate_start.cpp**: Demonstrates more advanced usage patterns
- **custom_ordering.cpp**: Shows how to use different ordering algorithms
- **custom_separator.cpp**: Demonstrates custom separator options
- **graph_node_change.cpp**: Shows handling of dynamic graph changes

Additionally, see [examples/cholesky_integration/](examples/cholesky_integration/) for integration with different solvers:

- **accelerate.cpp**: Integration with Apple's Accelerate framework
- **cholmod.cpp**: Integration with CHOLMOD solver
- **mkl.cpp**: Integration with Intel MKL

## API Reference

### Core Class: `PARTH::ParthAPI`

#### Configuration
- `setReorderingType(ReorderingType)`: Set ordering algorithm (METIS, AMD, MORTON_CODE)
- `setVerbose(bool)`: Enable/disable verbose output
- `setNDLevels(int)`: Set nested dissection levels
- `setNumberOfCores(int)`: Set number of cores for parallel processing

#### Mesh Input
- `setMesh(n, Mp, Mi)`: Set mesh connectivity in CSR format
- `setMesh(n, Mp, Mi, map)`: Set mesh with DOF mapping for dynamic meshes
- `setMatrix(N, Ap, Ai, dim)`: Set matrix connectivity and convert to mesh
- `setMatrix(N, Ap, Ai, map, dim)`: Set matrix with DOF mapping
- `setNewToOldDOFMap(map)`: Set mapping for factor reuse

#### Permutation Computation
- `computePermutation(perm, dim)`: Compute fill-reducing permutation
- `mapMeshPermToMatrixPerm(mesh_perm, matrix_perm, dim)`: Map to matrix DOFs

#### Analysis
- `getReuse()`: Get factor reuse percentage
- `getNumChanges()`: Get number of topology changes
- `printTiming()`: Print detailed timing information
- `resetTimers()`: Reset performance timers

---

## Dependencies

### Required
- **CMake** 3.14 or newer
- **C++17** compatible compiler (C++20 for the library itself)
- **Eigen3** (automatically downloaded if not found)
- **METIS** (required for the core functionality)

### Optional
- **Intel MKL** (for high-performance solvers)
- **CHOLMOD** (for CHOLMOD solver backend)
- **OpenMP** (for parallel processing)

### Build Options

Configure the build with these CMake options:

```bash
cmake -DPARTH_WITH_TESTS=ON \
      -DPARTH_WITH_API_DEMO=ON \
      -DPARTH_WITH_CHOLMOD_DEMO=ON \
      -DPARTH_WITH_ACCELERATE_DEMO=ON \
      -DPARTH_WITH_MKL_DEMO=ON \
      -DPARTH_WITH_SOLVER_WRAPPER=ON \
      ..
```

Available options:
- `PARTH_WITH_TESTS`: Build test suite (default: ON)
- `PARTH_WITH_API_DEMO`: Build API demonstration examples (default: OFF)
- `PARTH_WITH_CHOLMOD_DEMO`: Build CHOLMOD integration examples (default: OFF)  
- `PARTH_WITH_ACCELERATE_DEMO`: Build Accelerate integration examples (default: OFF)
- `PARTH_WITH_MKL_DEMO`: Build Intel MKL integration examples (default: OFF)
- `PARTH_WITH_SOLVER_WRAPPER`: Build solver wrapper utilities (default: OFF)

## Performance Tips

1. **Enable METIS**: Provides the best ordering quality for most problems
2. **Set appropriate ND levels**: Typically 4-8 levels work well
3. **Use factor reuse**: For dynamic meshes, provide DOF mapping to reuse previous factorizations
4. **Match problem dimension**: Set the correct DOF count per node (1 for scalar, 3 for 3D, etc.)

## Research Benchmarks

To reproduce the benchmarks from the Parth paper:

1. **Paper Examples**: For the original paper usage examples and benchmarks, switch to the `paper_code` branch
2. **IPC Benchmark**: See the [IPC benchmark generator repository](https://github.com/BehroozZare/parth-ipc-benchmark-generator.git) for matrix generation with Docker setup
3. **Remeshing Benchmark**: Download meshes from [Oded Stein's repository](https://github.com/odedstein/meshes) for testing with various mesh topologies

### Platform Notes

- **macOS**: Use Accelerate framework integration (recommended)
- **Linux**: All solvers (CHOLMOD, MKL, etc.) are supported
- **Docker**: Use the setup from the IPC benchmark repository for reproducible builds

---

## Citation

If you use Parth in your research or build upon this work, please cite:

```bibtex
@article{zarebavani2025adaptive,
  title={Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns},
  author={Zarebavani, Behrooz and Kaufman, Danny M. and Levin, David I.W. and Dehnavi, Maryam Mehri},
  journal={ACM Transactions on Graphics},
  volume={44},
  number={4},
  pages={1--17},
  year={2025},
  publisher={ACM},
  doi={10.1145/3687940},
  url={https://arxiv.org/pdf/2501.04011}
}
```

**Plain text citation:**

Behrooz Zarebavani, Danny M. Kaufman, David I.W. Levin, and Maryam Mehri Dehnavi. 2025. Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns. *ACM Transactions on Graphics* 44, 4 (August 2025), Article 123, 17 pages. https://doi.org/10.1145/3687940
