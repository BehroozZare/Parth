//
// Dynamic Mesh Demo - Shows how to use Parth with changing mesh topology
//
// This example demonstrates:
// 1. Initial mesh setup and permutation
// 2. Adding nodes to the mesh (mesh refinement)
// 3. Computing new permutation with reuse from previous ordering
// 4. Viewing reuse statistics
//

#include <parth/parth.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

class SimpleMesh {
public:
    std::vector<int> Mp, Mi;
    int n;
    
    void createInitialMesh() {
        // Create initial 3x3 grid (9 nodes)
        n = 9;
        Mp.resize(n + 1);
        Mi.clear();
        
        std::vector<std::vector<int>> adjacency(n);
        
        // Build adjacency for 3x3 grid
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int node = i * 3 + j;
                
                // Connect to neighbors
                if (i > 0) adjacency[node].push_back((i-1) * 3 + j); // up
                if (i < 2) adjacency[node].push_back((i+1) * 3 + j); // down
                if (j > 0) adjacency[node].push_back(i * 3 + (j-1)); // left
                if (j < 2) adjacency[node].push_back(i * 3 + (j+1)); // right
            }
        }
        
        buildCSR(adjacency);
        
        std::cout << "Created initial 3x3 mesh with " << n << " nodes" << std::endl;
    }
    
    void refineMesh() {
        // Add 4 more nodes to create a 4x4 grid (16 nodes total)
        // This simulates local mesh refinement
        
        int old_n = n;
        n = 16;
        
        std::vector<std::vector<int>> adjacency(n);
        
        // Copy old adjacency relationships
        for (int i = 0; i < old_n; i++) {
            for (int j = Mp[i]; j < Mp[i + 1]; j++) {
                adjacency[i].push_back(Mi[j]);
            }
        }
        
        // Build adjacency for 4x4 grid
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int node = i * 4 + j;
                if (node >= 16) break;
                
                adjacency[node].clear(); // Rebuild all connections
                
                // Connect to neighbors in 4x4 grid
                if (i > 0) adjacency[node].push_back((i-1) * 4 + j); // up
                if (i < 3) adjacency[node].push_back((i+1) * 4 + j); // down
                if (j > 0) adjacency[node].push_back(i * 4 + (j-1)); // left
                if (j < 3) adjacency[node].push_back(i * 4 + (j+1)); // right
            }
        }
        
        Mp.resize(n + 1);
        Mi.clear();
        buildCSR(adjacency);
        
        std::cout << "Refined mesh to 4x4 grid with " << n << " nodes" << std::endl;
    }
    
    std::vector<int> computeNewToOldMap() {
        // Create mapping from new mesh to old mesh
        std::vector<int> map(n);
        
        if (n == 9) {
            // Initial mesh - identity mapping
            for (int i = 0; i < n; i++) {
                map[i] = i;
            }
        } else if (n == 16) {
            // Refined mesh - map 4x4 grid positions to 3x3 positions where possible
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    int new_node = i * 4 + j;
                    
                    if (i < 3 && j < 3) {
                        // Existing node from 3x3 grid
                        int old_node = i * 3 + j;
                        map[new_node] = old_node;
                    } else {
                        // New node
                        map[new_node] = -1;
                    }
                }
            }
        }
        
        return map;
    }
    
private:
    void buildCSR(const std::vector<std::vector<int>>& adjacency) {
        Mp[0] = 0;
        for (int i = 0; i < n; i++) {
            Mp[i + 1] = Mp[i] + adjacency[i].size();
            for (int neighbor : adjacency[i]) {
                Mi.push_back(neighbor);
            }
        }
    }
};

void printPermutationSummary(const std::vector<int>& perm, const std::string& label) {
    std::cout << "\n" << label << " permutation:" << std::endl;
    std::cout << "Node:  ";
    for (int i = 0; i < std::min(10, static_cast<int>(perm.size())); i++) {
        std::cout << std::setw(4) << i;
    }
    std::cout << std::endl;
    std::cout << "Perm:  ";
    for (int i = 0; i < std::min(10, static_cast<int>(perm.size())); i++) {
        std::cout << std::setw(4) << perm[i];
    }
    std::cout << std::endl;
    if (perm.size() > 10) {
        std::cout << "(showing first 10 of " << perm.size() << " entries)" << std::endl;
    }
}

int main() {
    std::cout << "=== Parth Dynamic Mesh Demo ===" << std::endl;
    
    SimpleMesh mesh;
    PARTH::Parth parth;
    
    // Configure Parth
    parth.setReorderingType(PARTH::ReorderingType::METIS);
    parth.setVerbose(true);
    parth.setNDLevels(3);
    
    std::cout << "\n=== Step 1: Initial Mesh ===" << std::endl;
    
    // Create initial mesh
    mesh.createInitialMesh();
    
    // Set initial mesh data
    parth.setMeshPointers(mesh.n, mesh.Mp.data(), mesh.Mi.data());
    
    // Compute initial permutation
    std::vector<int> perm1;
    parth.computePermutation(perm1, 1);
    
    printPermutationSummary(perm1, "Initial");
    
    std::cout << "\nInitial timing:" << std::endl;
    parth.printTiming();
    
    std::cout << "\n=== Step 2: Mesh Refinement ===" << std::endl;
    
    // Refine the mesh
    mesh.refineMesh();
    
    // Create new-to-old mapping
    std::vector<int> new_to_old_map = mesh.computeNewToOldMap();
    
    std::cout << "New-to-old DOF mapping:" << std::endl;
    for (int i = 0; i < mesh.n; i++) {
        std::cout << "New node " << std::setw(2) << i << " -> ";
        if (new_to_old_map[i] == -1) {
            std::cout << "NEW" << std::endl;
        } else {
            std::cout << "old node " << new_to_old_map[i] << std::endl;
        }
    }
    
    // Reset timers to measure only the refinement step
    parth.resetTimers();
    
    // Set refined mesh with mapping
    parth.setMeshPointers(mesh.n, mesh.Mp.data(), mesh.Mi.data(), new_to_old_map);
    
    // Compute new permutation with reuse
    std::vector<int> perm2;
    parth.computePermutation(perm2, 1);
    
    printPermutationSummary(perm2, "Refined");
    
    std::cout << "\n=== Reuse Analysis ===" << std::endl;
    std::cout << "Factor reuse: " << std::fixed << std::setprecision(1) 
              << parth.getReuse() << "%" << std::endl;
    std::cout << "Number of changes: " << parth.getNumChanges() << std::endl;
    
    std::cout << "\nRefinement timing:" << std::endl;
    parth.printTiming();
    
    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "Original mesh: " << (mesh.n == 16 ? 9 : mesh.n) << " nodes" << std::endl;
    std::cout << "Refined mesh:  " << mesh.n << " nodes" << std::endl;
    std::cout << "New nodes:     " << std::count(new_to_old_map.begin(), new_to_old_map.end(), -1) << std::endl;
    
    std::cout << "\n=== Demo completed successfully! ===" << std::endl;
    return 0;
}
