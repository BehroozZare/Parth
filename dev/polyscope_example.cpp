//
// Example demonstrating Polyscope integration with Parth
//

#include "utils/polyscope_utils.h"
#include "utils/vertex_colored_mesh.h"
#include <Eigen/Core>
#include <iostream>

int main() {
    std::cout << "Parth Polyscope Integration Example" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Initialize polyscope
    PARTH::utils::polyscope_utils::initialize();
    
    // Create a simple triangle mesh
    Eigen::MatrixXd vertices(3, 3);
    vertices << 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                0.5, 1.0, 0.0;
    
    Eigen::MatrixXi faces(1, 3);
    faces << 0, 1, 2;
    
    // Register the mesh with polyscope
    PARTH::utils::polyscope_utils::registerSurfaceMesh("triangle", vertices, faces);
    
    // Create some vertex colors (red, green, blue)
    Eigen::MatrixXd colors(3, 3);
    colors << 1.0, 0.0, 0.0,  // Red
              0.0, 1.0, 0.0,  // Green
              0.0, 0.0, 1.0;  // Blue
    
    // Add vertex colors
    PARTH::utils::polyscope_utils::addVertexColors("triangle", "vertex_colors", colors);
    
    // Create some scalar values
    Eigen::VectorXd scalars(3);
    scalars << 0.0, 0.5, 1.0;
    
    // Add vertex scalars
    PARTH::utils::polyscope_utils::addVertexScalars("triangle", "height", scalars);
    
    // Test the VertexColoredMesh class
    std::cout << "\nTesting VertexColoredMesh class:" << std::endl;
    PARTH::utils::VertexColoredMesh mesh;
    mesh.setMesh(vertices, faces);
    mesh.setVertexColors(colors);
    mesh.setVertexScalars(scalars);
    mesh.visualize("test_mesh");
    
    std::cout << "\nMesh setup complete. Call polyscope_utils::show() to display the GUI." << std::endl;
    
    // Uncomment the next line to show the polyscope GUI
    PARTH::utils::polyscope_utils::show();
    
    return 0;
}
