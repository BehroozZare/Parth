//
// Created by behrooz on 19/10/22.
//
//
// Created by behrooz on 11/10/22.
//
//
// Created by behrooz on 28/09/22.
//

#include "NewtonIterVisualization.h"
#include "Parth_utils.h"
#include "Solver.h"
#include "cholmod.h"
#include <CLI/CLI.hpp>
#include <Eigen/Eigen>
#include <metis.h>
#include <unsupported/Eigen/SparseExtra>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

struct CLIArgs {
  std::string input_address =
      "/media/LargeHard/IPC/test_out/"
      "dolphin5K_null_NH_BE_interiorPoint_20221019205148/";
  std::string output_address = "../output";
  std::string ordMethod = "METIS";

  bool MetisK = true;
  std::string mat_type = "mesh";
  int synthesize_mat_size = 50;

  CLIArgs(int argc, char *argv[]) {
    CLI::App app{"Parth Solver"};

    // Custom flags
    app.add_option("--input", input_address, "Input address");
    app.add_option("--MetisK", MetisK, "Size of the synthesize matrix");
    app.add_option("--MatType", mat_type, "Size of the synthesize matrix");

    app.add_option("--OrderingAlg", ordMethod, "Ordering Name");

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      exit(app.exit(e));
    }
  }
};

int main(int argc, char *argv[]) {
  CLIArgs args(argc, argv);

  //  std::string mat_type = "hessian";
  std::string path = args.input_address;
  auto list_of_files = PARTH::getFileNames(path);
  std::vector<std::tuple<int, int>> list_of_iter;
  for (auto &x : list_of_files) {
    auto s = PARTH::split_string(x, "_");
    if (s.empty() || s[0] != args.mat_type) {
      continue;
    }
    list_of_iter.emplace_back(
        std::tuple<int, int>(std::stoi(s[1]), std::stoi(s[2])));
  }

  auto cholmod_solver =
      PARTH::Solver::create(PARTH::LinSysSolverType::CHOLMOD_Profile);

  auto parth_solver = PARTH::Solver::create(PARTH::LinSysSolverType::LL_PARTH);

  int max_iterations = 5;
  int cnt = 0;
  // Main loop for analyzing iterations of a frame
  for (auto &iter : list_of_iter) {
    if (cnt == max_iterations) {
      break;
    }
    cnt++;

    Eigen::SparseMatrix<double> lower_A_csc;
    Eigen::SparseMatrix<double> lower_mesh_csc;
    std::string hessian_name =
        path + "hessian_" + std::to_string(std::get<0>(iter)) + "_" +
        std::to_string(std::get<1>(iter)) + "_" + "IPC.mtx";
    //************** Load the hessian *************
    if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
      std::cerr << "File " << hessian_name << " is not found" << std::endl;
    }
    //************** Load the Mesh *************
    std::string mesh_name = path + "mesh_" + std::to_string(std::get<0>(iter)) +
                            "_" + std::to_string(std::get<1>(iter)) + "_" +
                            "IPC.mtx";
    if (!Eigen::loadMarket(lower_mesh_csc, mesh_name)) {
      std::cerr << "File " << hessian_name << " is not found" << std::endl;
    }
    //***********************************************************************
    Eigen::VectorXd x_eigen(lower_A_csc.rows());
    Eigen::VectorXd b_eigen(lower_A_csc.rows());
    x_eigen.setZero();
    b_eigen.setOnes();

    cholmod_solver->setMatrix(lower_A_csc);
    cholmod_solver->analyze();
    cholmod_solver->factorize();
    cholmod_solver->solve(b_eigen, x_eigen);

    std::cout << "Number of nodes: " << cholmod_solver->getNumRows()
              << " - A #nonzeros: " << cholmod_solver->getNumNonzeros()
              << " - L #nonzeros: " << cholmod_solver->getFactorNonzeros()
              << " - (L #nonzeros / A #nonzeros): "
              << cholmod_solver->getFactorNonzeros() /
                     cholmod_solver->getNumNonzeros()
              << std::endl;

    std::cout << "Profile: reordering time is: "
              << cholmod_solver->getAnalyzeTime() << std::endl;
    std::cout << "Profile: factorization time is: "
              << cholmod_solver->getFactorTime() << std::endl;
    std::cout << "Profile: solve time is: " << cholmod_solver->getSolveTime()
              << std::endl;
    std::cout << "Profile: Res is:" << cholmod_solver->getResidual()
              << std::endl;

    std::cout << "Profile: The average supernode size is:"
              << cholmod_solver->inspector().getAvgSuperNodeSize() << std::endl;

    //***********************************************************************
    x_eigen.setZero();
    b_eigen.setOnes();

    parth_solver->setMatrix(lower_A_csc);
    parth_solver->setMesh(lower_mesh_csc);
    parth_solver->analyze();
    parth_solver->factorize();
    parth_solver->solve(b_eigen, x_eigen);

    std::cout << "Number of nodes: " << parth_solver->getNumRows()
              << " - A #nonzeros: " << parth_solver->getNumNonzeros()
              << " - L #nonzeros: " << parth_solver->getFactorNonzeros()
              << " - (L #nonzeros / A #nonzeros): "
              << parth_solver->getFactorNonzeros() /
                     parth_solver->getNumNonzeros()
              << std::endl;

    std::cout << "PARTH: reordering time is: " << parth_solver->getAnalyzeTime()
              << std::endl;
    std::cout << "PARTH: factorization time is: "
              << parth_solver->getFactorTime() << std::endl;
    std::cout << "PARTH: solve time is: " << parth_solver->getSolveTime()
              << std::endl;
    std::cout << "PARTH: Res is:" << parth_solver->getResidual() << std::endl;

    std::cout << "PARTH: The average supernode size is:"
              << parth_solver->inspector().getAvgSuperNodeSize() << std::endl;
  }

  delete cholmod_solver;
  delete parth_solver;

  return 0;
}
