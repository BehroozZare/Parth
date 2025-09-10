//
// Created for Parth project vertex coloring utilities
//

#include "vertex_colored_mesh.h"
#include <iostream>

namespace PARTH {
namespace utils {

void VertexColoredMesh::setMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces) {
    vertices_ = vertices;
    faces_ = faces;
    has_vertices_ = true;
    has_faces_ = true;
}

void VertexColoredMesh::setVertexColors(const Eigen::MatrixXd& colors) {
    if (!has_vertices_) {
        std::cerr << "Warning: Setting colors before vertices. Please set mesh first." << std::endl;
        return;
    }
    
    if (colors.rows() != vertices_.rows()) {
        std::cerr << "Error: Number of colors must match number of vertices." << std::endl;
        return;
    }
    
    vertex_colors_ = colors;
    has_colors_ = true;
}

void VertexColoredMesh::setVertexScalars(const Eigen::VectorXd& scalars) {
    if (!has_vertices_) {
        std::cerr << "Warning: Setting scalars before vertices. Please set mesh first." << std::endl;
        return;
    }
    
    if (scalars.rows() != vertices_.rows()) {
        std::cerr << "Error: Number of scalars must match number of vertices." << std::endl;
        return;
    }
    
    vertex_scalars_ = scalars;
    has_scalars_ = true;
}

void VertexColoredMesh::visualize(const std::string& name) const {
    if (!has_vertices_ || !has_faces_) {
        std::cerr << "Error: Cannot visualize mesh without vertices and faces." << std::endl;
        return;
    }
    
    std::cout << "Note: Polyscope visualization functionality requires linking with polyscope library." << std::endl;
    std::cout << "Mesh '" << name << "' data is ready for visualization:" << std::endl;
    std::cout << "  - Vertices: " << vertices_.rows() << " x " << vertices_.cols() << std::endl;
    std::cout << "  - Faces: " << faces_.rows() << " x " << faces_.cols() << std::endl;
    
    if (has_colors_) {
        std::cout << "  - Has vertex colors: " << vertex_colors_.rows() << " x " << vertex_colors_.cols() << std::endl;
    }
    if (has_scalars_) {
        std::cout << "  - Has vertex scalars: " << vertex_scalars_.rows() << std::endl;
    }
}

} // namespace utils
} // namespace PARTH
