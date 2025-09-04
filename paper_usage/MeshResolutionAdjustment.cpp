//
// Created by behrooz zare on 2024-04-28.
//
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/avg_edge_length.h>
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <igl/upsample.h>
#include <remesh_botsch.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

int main(int argc, char *argv[])
{
  // Initialize the viewer
  igl::opengl::glfw::Viewer viewer;

  // Check if a file is provided as an argument
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <path/to/mesh.obj>" << std::endl;
    return -1;
  }

  // Load a triangle mesh (square sheet)
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  if (!igl::read_triangle_mesh(argv[1], V, F))
  {
    std::cerr << "Error reading mesh from " << argv[1] << std::endl;
    return -1;
  }
  //Make the y axis of V equal to zero and then copy z axis to y axis
//  V.col(1).setZero();
//  V.col(1) = V.col(2);
//  V.col(2).setZero();
  // Find boundary edges
   igl::upsample(V, F, V, F, 2);
//
  // Print the rotated mesh vertices (optional)
  std::cout << "Rotated vertices:\n" << V << std::endl;
  //Save The current obj file with new V and F
  igl::writeOBJ("/home/behrooz/Desktop/IPC_Project/ParthSolverDev/data/square_2.obj", V, F);

  // Set the viewer data
  viewer.data().set_mesh(V, F);

  // Launch the viewer
  viewer.launch();

  return 0;
}
