# Parth: Fill-Reducing Orderings for Sparse Cholesky Factorization

[![Ubuntu](https://img.shields.io/github/actions/workflow/status/BehroozZare/Parth/build.yml?branch=main&label=Ubuntu&logo=ubuntu&logoColor=white)](https://github.com/BehroozZare/Parth/actions/workflows/build.yml)
[![macOS](https://img.shields.io/github/actions/workflow/status/BehroozZare/Parth/build.yml?branch=main&label=macOS&logo=apple&logoColor=white)](https://github.com/BehroozZare/Parth/actions/workflows/build.yml)
[![Windows](https://img.shields.io/github/actions/workflow/status/BehroozZare/Parth/build.yml?branch=main&label=Windows&logo=windows&logoColor=white)](https://github.com/BehroozZare/Parth/actions/workflows/build.yml)

Parth accelerates sparse Cholesky factorizations by providing fill-reducing orderings that **reuse computations** when sparsity patterns change dynamically. Works with MKL, Accelerate, CHOLMOD, and other state-of-the-art solvers.

## ⚡ 30-Second Example

```cpp
#include <parth/parth.h>

PARTH::ParthAPI parth;
parth.setMatrix(n, column_ptrs, row_indices, 1);  // Set your sparse matrix
std::vector<int> perm;
parth.computePermutation(perm);  // Get fill-reducing permutation
// Use perm with your favorite sparse solver (MKL, CHOLMOD, etc.)
```

**Expected output**: Permutation vector 

## 🚀 Three Ways to Get Started

### 1. 📚 Learn the Basics
**See it in action first**: [`examples/api_demos/quick_start.cpp`](examples/api_demos/quick_start.cpp)
- Complete working example showing computational reuse
- Shows the core Parth workflow step-by-step

### 2. 🔧 Integrate with Your Solver  
**Ready-to-use solver integrations**: [`examples/cholesky_integration/`](examples/cholesky_integration/)
- **MKL**: [`mkl.cpp`](examples/cholesky_integration/mkl.cpp) - Intel MKL PARDISO integration
- **Accelerate**: [`accelerate.cpp`](examples/cholesky_integration/accelerate.cpp) - Apple Accelerate framework  
- **CHOLMOD**: [`cholmod.cpp`](examples/cholesky_integration/cholmod.cpp) - SuiteSparse CHOLMOD

### 3. 🚀 Quick CMake Integration
**Copy-paste project setup**: [`integration_example/`](integration_example/)
- Complete CMake project template
- Automatic dependency fetching with FetchContent
- One-command build: `./scripts/build_with_parth.sh`

---

## Installation

Add to your `CMakeLists.txt`:

```cmake
include(FetchContent)
FetchContent_Declare(parth
    GIT_REPOSITORY https://github.com/BehroozZare/parth.git
    GIT_TAG main)
FetchContent_MakeAvailable(parth)
target_link_libraries(your_target PRIVATE Parth::parth)
```

## Basic Usage

```cpp
#include <parth/parth.h>

PARTH::ParthAPI parth;
parth.setMatrix(n, column_ptrs, row_indices, 1);  // CSR/CSC sparse matrix
std::vector<int> perm;
parth.computePermutation(perm);  // Get permutation

// Use permutation with your favorite sparse solver...
```

## ✅ Verify Installation

Test that everything works:

```bash
# Clone and test the integration example
git clone https://github.com/BehroozZare/parth.git
cd parth/integration_example
./scripts/build_with_parth.sh && cd build && ./use_parth
```

**Expected output**: 
```
Computing permutation (from scratch)...
Computing permutation (with reuse from previous computation)...
Modified matrix permutation computed successfully.
```

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
- **Intel MKL** (for high-performance sparse solvers for intel processors)
- **Apple Accelerate** (for high-performance sparse solvers in Apple silicon)

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
2. **Use factor reuse**: For dynamic meshes (meshes with change in number DOFs), provide DOF mapping to reuse previous factorizations
3. **Match problem dimension**: Set the correct DOF count per node (1 for scalar, 3 for 3D, etc.)

## Research Benchmarks

To reproduce the benchmarks from the Parth paper:

1. **Paper Examples**: For the original paper usage examples and benchmarks, switch to the `paper_code` branch
2. **IPC Benchmark**: See the [IPC benchmark generator repository](https://github.com/BehroozZare/parth-ipc-benchmark-generator.git) for matrix generation with Docker setup
3. **Remeshing Benchmark**: Download meshes from [Oded Stein's repository](https://github.com/odedstein/meshes) for testing with various mesh topologies

### Platform Notes

- **macOS**: Use Accelerate framework integration (recommended)
- **Linux**: All solvers (CHOLMOD, MKL, etc.) are supported
- **Docker**: Use the setup from [this repository](https://github.com/BehroozZare/parth-ipc-benchmark-generator.git) for reproducible builds

---

## Citation

If you use Parth in your research or build upon this work, please cite:

```bibtex
@article{zarebavani2025adaptive,
  title={Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns},
  author={Zarebavani, Behrooz and M. Kaufman, Danny and IW Levin, David and Mehri Dehnavi, Maryam},
  journal={ACM Transactions on Graphics (TOG)},
  volume={44},
  number={4},
  pages={1--17},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```