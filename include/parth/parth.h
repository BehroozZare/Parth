//
// Parth: Fill-reducing orderings for sparse Cholesky factorization
// 
// Public API header for the Parth library.
// 
// This header provides access to the Parth class for computing fill-reducing
// orderings that can be used with sparse Cholesky solvers.
//
// Basic usage:
//   #include <parth/parth.h>
//
//   PARTH::Parth parth;
//   parth.setReorderingType(PARTH::ReorderingType::METIS);
//   parth.setMeshPointers(n, Mp, Mi);
//   
//   std::vector<int> perm;
//   parth.computePermutation(perm, 3); // 3 DOFs per node
//

#ifndef PARTH_PUBLIC_H
#define PARTH_PUBLIC_H

// Include the actual implementation headers
// The CMake build system will set up include paths so these resolve correctly
#include "Parth.h"         // Implementation class
#include "ParthTypes.h"    // Enum definitions

// All types and classes are already in the PARTH namespace from the implementation
// No need to re-export anything - just include and use directly

#endif // PARTH_PUBLIC_H
