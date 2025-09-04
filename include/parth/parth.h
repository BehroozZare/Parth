//
// Parth: Fill-reducing orderings for sparse Cholesky factorization
// 
// This is the main public API header for the Parth library.
//

#ifndef PARTH_H
#define PARTH_H

#include <vector>

namespace PARTH {

/// Reordering algorithm types
enum class ReorderingType { 
    METIS,      ///< Use METIS ordering
    AMD,        ///< Use AMD ordering  
    AUTO        ///< Automatically select best ordering
};

/// Main Parth class for computing fill-reducing orderings
class Parth {
public:
    /// Constructor
    Parth();
    
    /// Destructor
    ~Parth();

    //============================ CONFIGURATION =================================
    
    /// Set the reordering algorithm type
    /// @param type The reordering algorithm to use
    void setReorderingType(ReorderingType type);
    
    /// Get the current reordering algorithm type
    /// @return The current reordering type
    ReorderingType getReorderingType() const;

    /// Set verbose output (print timing and debug information)
    /// @param verbose If true, print detailed information
    void setVerbose(bool verbose);
    
    /// Get verbose setting
    /// @return Current verbose setting
    bool getVerbose() const;

    /// Set number of nested dissection levels for hierarchical mesh decomposition
    /// @param levels Number of ND levels (typically 4-8)
    void setNDLevels(int levels);
    
    /// Get number of nested dissection levels
    /// @return Current ND levels
    int getNDLevels() const;

    /// Set number of cores for parallel processing
    /// @param num_cores Number of cores to use
    void setNumberOfCores(int num_cores);
    
    /// Get number of cores
    /// @return Current number of cores
    int getNumberOfCores() const;

    //============================ MESH INPUT ===================================
    
    /// Set mesh connectivity (for first time or when topology doesn't change)
    /// @param n Number of mesh nodes
    /// @param Mp CSR format pointer array (size n+1)
    /// @param Mi CSR format index array
    void setMeshPointers(int n, int* Mp, int* Mi);
    
    /// Set mesh connectivity with DOF mapping (for dynamic meshes)
    /// @param n Number of mesh nodes  
    /// @param Mp CSR format pointer array (size n+1)
    /// @param Mi CSR format index array
    /// @param new_to_old_map Maps new DOF indices to old ones (-1 for new DOFs)
    void setMeshPointers(int n, int* Mp, int* Mi, const std::vector<int>& new_to_old_map);

    /// Set mapping from new DOF indices to old DOF indices
    /// @param map Vector where map[i] = old_index of new_index i, or -1 if new
    void setNewToOldDOFMap(const std::vector<int>& map);

    //============================ PERMUTATION COMPUTATION ======================
    
    /// Compute fill-reducing permutation for the mesh
    /// @param perm Output permutation vector (will be resized)
    /// @param dim Problem dimension (1 for scalar, 3 for 3D elasticity, etc.)
    void computePermutation(std::vector<int>& perm, int dim = 3);

    /// Map mesh permutation to matrix permutation for multi-DOF problems
    /// @param mesh_perm Input mesh-level permutation  
    /// @param matrix_perm Output matrix-level permutation (will be resized)
    /// @param dim Number of DOFs per mesh node
    void mapMeshPermToMatrixPerm(const std::vector<int>& mesh_perm, 
                                 std::vector<int>& matrix_perm, int dim = 3);

    //============================ ANALYSIS =====================================
    
    /// Get factor reuse percentage from previous factorization
    /// @return Percentage of factors that can be reused (0-100)
    double getReuse() const;
    
    /// Get number of changed connections since last ordering
    /// @return Number of changed mesh connections
    int getNumChanges() const;
    
    /// Print detailed timing information
    void printTiming() const;
    
    /// Reset all internal timers
    void resetTimers();

    /// Clear all internal data structures
    void clearParth();

private:
    class Impl;
    Impl* pImpl;
};

} // namespace PARTH

#endif // PARTH_H
