//
// Created for Parth project vertex coloring utilities
//

#ifndef PARTH_VERTEX_COLORED_MESH_H
#define PARTH_VERTEX_COLORED_MESH_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace PARTH {
namespace utils {

/// Utility class for handling vertex colored meshes
class VertexColoredMesh {
public:
    /// Constructor
    VertexColoredMesh() = default;
    
    /// Destructor
    ~VertexColoredMesh() = default;
    
    /// Set mesh vertices and faces
    /// \param vertices Nx3 matrix of vertex positions
    /// \param faces Fx3 matrix of face indices
    void setMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);
    
    /// Set vertex colors
    /// \param colors Nx3 matrix of RGB colors (values between 0 and 1)
    void setVertexColors(const Eigen::MatrixXd& colors);
    
    /// Set vertex scalar values for color mapping
    /// \param scalars Nx1 vector of scalar values
    void setVertexScalars(const Eigen::VectorXd& scalars);
    
    /// Visualize the mesh using Polyscope
    /// \param name Name of the mesh in Polyscope
    void visualize(const std::string& name = "mesh") const;
    
    /// Get vertices
    const Eigen::MatrixXd& getVertices() const { return vertices_; }
    
    /// Get faces
    const Eigen::MatrixXi& getFaces() const { return faces_; }
    
    /// Get vertex colors
    const Eigen::MatrixXd& getVertexColors() const { return vertex_colors_; }
    
    /// Get vertex scalars
    const Eigen::VectorXd& getVertexScalars() const { return vertex_scalars_; }

private:
    Eigen::MatrixXd vertices_;      ///< Vertex positions (Nx3)
    Eigen::MatrixXi faces_;         ///< Face indices (Fx3)
    Eigen::MatrixXd vertex_colors_; ///< Vertex colors (Nx3)
    Eigen::VectorXd vertex_scalars_; ///< Vertex scalar values (Nx1)
    
    bool has_vertices_ = false;
    bool has_faces_ = false;
    bool has_colors_ = false;
    bool has_scalars_ = false;
};

} // namespace utils
} // namespace PARTH

#endif // PARTH_VERTEX_COLORED_MESH_H
