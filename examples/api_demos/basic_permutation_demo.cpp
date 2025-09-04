//
// Basic Permutation Demo - Shows how to use Parth to compute mesh permutations
//
// This example demonstrates:
// 1. Setting up a simple mesh 
// 2. Computing a permutation using Parth
// 3. Viewing the results
//

#include <parth/parth.h>
#include <iostream>
#include <vector>
#include <iomanip>

void createSimple2DMesh(std::vector<int>& Mp, std::vector<int>& Mi, int& n) {
    // Create a simple 4x4 grid mesh (25 nodes, 16 elements)
    // Each element connects to 4 adjacent nodes
    
    n = 25; // 5x5 grid of nodes
    Mp.resize(n + 1);
    
    std::vector<std::vector<int>> adjacency(n);
    
    // Build adjacency for 5x5 grid
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            int node = i * 5 + j;
            
            // Connect to neighbors
            if (i > 0) adjacency[node].push_back((i-1) * 5 + j); // up
            if (i < 4) adjacency[node].push_back((i+1) * 5 + j); // down
            if (j > 0) adjacency[node].push_back(i * 5 + (j-1)); // left
            if (j < 4) adjacency[node].push_back(i * 5 + (j+1)); // right
        }
    }
    
    // Convert to CSR format
    Mp[0] = 0;
    for (int i = 0; i < n; i++) {
        Mp[i + 1] = Mp[i] + adjacency[i].size();
        for (int neighbor : adjacency[i]) {
            Mi.push_back(neighbor);
        }
    }
    
    std::cout << "Created 5x5 grid mesh with " << n << " nodes and " 
              << Mi.size() << " connections" << std::endl;
}

int main() {
    std::cout << "=== Parth Basic Permutation Demo ===" << std::endl;
    
    // Create a simple mesh
    std::vector<int> Mp, Mi;
    int n;
    createSimple2DMesh(Mp, Mi, n);
    
    // Initialize Parth
    PARTH::Parth parth;
    
    // Configure Parth
    parth.setReorderingType(PARTH::ReorderingType::METIS);
    parth.setVerbose(true);
    parth.setNDLevels(3);  // Fewer levels for small mesh
    
    std::cout << "\nParth Configuration:" << std::endl;
    std::cout << "- Reordering: METIS" << std::endl;
    std::cout << "- ND Levels: " << parth.getNDLevels() << std::endl;
    std::cout << "- Verbose: " << (parth.getVerbose() ? "enabled" : "disabled") << std::endl;
    
    // Set mesh data
    parth.setMeshPointers(n, Mp.data(), Mi.data());
    
    // Compute permutation
    std::cout << "\nComputing permutation..." << std::endl;
    std::vector<int> perm;
    parth.computePermutation(perm, 1); // 1 DOF per node (scalar problem)
    
    // Display results
    std::cout << "\nOriginal node order vs. Permuted order:" << std::endl;
    std::cout << "Original -> Permuted" << std::endl;
    for (int i = 0; i < std::min(15, n); i++) { // Show first 15 nodes
        std::cout << std::setw(6) << i << " -> " << std::setw(6) << perm[i] << std::endl;
    }
    if (n > 15) {
        std::cout << "... (showing first 15 of " << n << " nodes)" << std::endl;
    }
    
    // Show timing information
    std::cout << "\nTiming Information:" << std::endl;
    parth.printTiming();
    
    // For multi-DOF problems, show how to map to matrix permutation
    std::cout << "\n=== Multi-DOF Example (3 DOFs per node) ===" << std::endl;
    std::vector<int> matrix_perm;
    parth.mapMeshPermToMatrixPerm(perm, matrix_perm, 3);
    
    std::cout << "Matrix size: " << matrix_perm.size() << " DOFs" << std::endl;
    std::cout << "First few matrix permutations (node_id * 3 + dof_offset):" << std::endl;
    for (int i = 0; i < std::min(15, static_cast<int>(matrix_perm.size())); i++) {
        std::cout << "DOF " << std::setw(2) << i << " -> DOF " << std::setw(2) << matrix_perm[i] << std::endl;
    }
    
    std::cout << "\n=== Demo completed successfully! ===" << std::endl;
    return 0;
}
