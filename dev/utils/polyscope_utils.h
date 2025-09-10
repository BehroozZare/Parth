//
// Created for Parth project polyscope integration utilities
//

#ifndef PARTH_POLYSCOPE_UTILS_H
#define PARTH_POLYSCOPE_UTILS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>

namespace PARTH {
namespace utils {

/// Utility functions for Polyscope integration
namespace polyscope_utils {

/// Initialize Polyscope with default settings
void initialize();

/// Show the Polyscope GUI
void show();

/// Save a screenshot of the current visualization
/// \param filename Path and filename for the saved image
/// \param transparentBG Whether to use transparent background (default: true)
void saveScreenshot(const std::string& filename, bool transparentBG = true);

/// Register a surface mesh with Polyscope
/// \param name Name of the mesh
/// \param vertices Nx3 matrix of vertex positions
/// \param faces Fx3 matrix of face indices
void registerSurfaceMesh(const std::string& name, 
                        const Eigen::MatrixXd& vertices, 
                        const Eigen::MatrixXi& faces);

/// Add vertex colors to a registered mesh
/// \param mesh_name Name of the registered mesh
/// \param quantity_name Name for the color quantity
/// \param colors Nx3 matrix of RGB colors
void addVertexColors(const std::string& mesh_name,
                    const std::string& quantity_name,
                    const Eigen::MatrixXd& colors);

/// Add vertex scalars to a registered mesh
/// \param mesh_name Name of the registered mesh
/// \param quantity_name Name for the scalar quantity
/// \param scalars Nx1 vector of scalar values
void addVertexScalars(const std::string& mesh_name,
                     const std::string& quantity_name,
                     const Eigen::VectorXd& scalars);

} // namespace polyscope_utils
} // namespace utils
} // namespace PARTH

#endif // PARTH_POLYSCOPE_UTILS_H
