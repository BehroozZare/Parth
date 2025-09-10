//
// Created for Parth project polyscope integration utilities
//

#include "polyscope_utils.h"
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/screenshot.h>
#include <iostream>

namespace PARTH {
namespace utils {
namespace polyscope_utils {

void initialize() {
    polyscope::init();
    std::cout << "Polyscope initialized successfully." << std::endl;
}

void show() {
    polyscope::show();
}

void saveScreenshot(const std::string& filename, bool transparentBG) {
    polyscope::screenshot(filename, transparentBG);
    std::cout << "Screenshot saved to: " << filename << std::endl;
}

void registerSurfaceMesh(const std::string& name, 
                        const Eigen::MatrixXd& vertices, 
                        const Eigen::MatrixXi& faces) {
    if (vertices.cols() != 3) {
        std::cerr << "Error: Vertices must be Nx3 matrix." << std::endl;
        return;
    }
    if (faces.cols() != 3) {
        std::cerr << "Error: Faces must be Fx3 matrix." << std::endl;
        return;
    }
    
    polyscope::registerSurfaceMesh(name, vertices, faces);
    std::cout << "Registered surface mesh '" << name << "' with " 
              << vertices.rows() << " vertices and " << faces.rows() << " faces." << std::endl;
}

void addVertexColors(const std::string& mesh_name,
                    const std::string& quantity_name,
                    const Eigen::MatrixXd& colors) {
    auto mesh = polyscope::getSurfaceMesh(mesh_name);
    if (!mesh) {
        std::cerr << "Error: Mesh '" << mesh_name << "' not found." << std::endl;
        return;
    }
    
    if (colors.cols() != 3) {
        std::cerr << "Error: Colors must be Nx3 matrix." << std::endl;
        return;
    }
    
    auto colorQuantity = mesh->addVertexColorQuantity(quantity_name, colors);
    colorQuantity->setEnabled(true);  // Enable the color quantity by default
    std::cout << "Added and enabled vertex colors '" << quantity_name << "' to mesh '" << mesh_name << "'." << std::endl;
}

void addVertexScalars(const std::string& mesh_name,
                     const std::string& quantity_name,
                     const Eigen::VectorXd& scalars) {
    auto mesh = polyscope::getSurfaceMesh(mesh_name);
    if (!mesh) {
        std::cerr << "Error: Mesh '" << mesh_name << "' not found." << std::endl;
        return;
    }
    
    mesh->addVertexScalarQuantity(quantity_name, scalars);
    std::cout << "Added vertex scalars '" << quantity_name << "' to mesh '" << mesh_name << "'." << std::endl;
}

} // namespace polyscope_utils
} // namespace utils
} // namespace PARTH
