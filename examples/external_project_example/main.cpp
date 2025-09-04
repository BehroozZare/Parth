//
// Example main.cpp showing how to use Parth in an external project
//

#include <parth/parth.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== External Project Using Parth ===" << std::endl;
    
    // Create a simple test mesh (triangle mesh)
    int n = 6; // 6 nodes forming 2 triangles
    std::vector<int> Mp = {0, 2, 4, 6, 8, 10, 12}; // CSR pointers
    std::vector<int> Mi = {1, 3, 0, 2, 1, 4, 0, 5, 2, 5, 3, 4}; // CSR indices
    
    std::cout << "Created simple mesh with " << n << " nodes" << std::endl;
    
    // Initialize Parth
    PARTH::Parth parth;
    parth.setReorderingType(PARTH::ReorderingType::METIS);
    parth.setVerbose(true);
    
    // Set mesh data
    parth.setMeshPointers(n, Mp.data(), Mi.data());
    
    // Compute permutation
    std::vector<int> perm;
    parth.computePermutation(perm, 1); // 1 DOF per node
    
    // Display results
    std::cout << "\nComputed permutation:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "Node " << i << " -> " << perm[i] << std::endl;
    }
    
    // Show timing
    parth.printTiming();
    
    std::cout << "\n=== Success! Parth integration working correctly ===" << std::endl;
    return 0;
}
