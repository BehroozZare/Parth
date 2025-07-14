//
// Created by behrooz zare on 2024-04-07.
//

#include "Parth.h"
#include "ParthTestUtils.h"
#include <Eigen/Eigen>
#include <iostream>
#include <string>
#include <unsupported/Eigen/SparseExtra>

int main(int argc, char *argv[]) {

  //************** Load the Mesh  *************
  std::string start_address =
      "/home/behrooz/Desktop/IPC_Project/SolverTestBench/data/IPC_mesh/";
  Eigen::SparseMatrix<double> mesh_csc;
  std::string mesh_name = start_address + "mesh_0_1_IPC.mtx";
  std::string hessian_name = start_address + "hessian_0_1_last_IPC.mtx";

  //************** Setup the tester *************
  PARTH::ParthTestUtils tester;

  //  //************** Test the adding and deleting Edges *************
  //  tester.testAddingDeletingEdges(hessian_name, mesh_name);

  //************** Test the adding and deleting DOFs *************
  tester.testAddingDeletingDOFs(hessian_name, mesh_name);

  //************** Test the permutation quality *************
  //  tester.testPermutationQuality(hessian_name, mesh_name);

  return 0;
}
