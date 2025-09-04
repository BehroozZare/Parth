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
  // Sort the list of iterations
  std::sort(list_of_iter.begin(), list_of_iter.end(), PARTH::sortbyFirstAndSec);

  // Load the SF matrix
  Eigen::MatrixXi SF = PARTH::openMatInt(path + "SF");
  // Load the V_rest matrix
  Eigen::MatrixXd V_rest = PARTH::openMatDouble(path + "V_rest");

  auto parth_solver = PARTH::Solver::create(PARTH::LinSysSolverType::LL_PARTH);

  // Initializing the iter view class
  PARTH::NewtonIterVisualization vis_obj = PARTH::NewtonIterVisualization();
  vis_obj.init("arma", true,
               "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/");

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
    //************** Load the positions *************
    Eigen::MatrixXd V =
        PARTH::openMatDouble(path + "V_" + std::to_string(std::get<0>(iter)) +
                             "_" + std::to_string(std::get<1>(iter)));
    std::cout << "Number of node in a mesh " << V.rows() << std::endl;
    std::cout << "Number of rows in the hessian " << lower_A_csc.rows()
              << std::endl;

    //***********************************************************************
    Eigen::VectorXd x_eigen(lower_A_csc.rows());
    Eigen::VectorXd b_eigen(lower_A_csc.rows());
    x_eigen.setZero();
    b_eigen.setOnes();

    parth_solver->setMatrix(lower_A_csc);
    parth_solver->setMesh(lower_mesh_csc);
    parth_solver->analyze();

    std::cout << "Profile: reordering time is: "
              << parth_solver->getAnalyzeTime() << std::endl;

    std::vector<int> part;
    if (args.MetisK) {
      Eigen::SparseMatrix<double> full_A =
          lower_A_csc.selfadjointView<Eigen::Lower>();
      idx_t nVertices = full_A.rows();
      idx_t nWeights = 1;
      idx_t nParts = 2;
      idx_t objval;
      part.resize(nVertices, 0);
      std::vector<int> vweight(nVertices, 1);
      double ratio = 0.8;
      int part_v = nVertices * ratio;
      double weight_ratio = (1 - ratio) / ratio;
      if (weight_ratio >= 1) {
        int chosen_weight = weight_ratio * 1000;
        std::fill(vweight.data(), vweight.data() + part_v, chosen_weight);
      } else {
        int chosen_weight = (1 / weight_ratio) * 1000;
        std::fill(vweight.data() + part_v, vweight.data() + vweight.size(),
                  chosen_weight);
      }

      int ret = METIS_PartGraphKway(
          &nVertices, &nWeights, full_A.outerIndexPtr(), full_A.innerIndexPtr(),
          vweight.data(), &nVertices, NULL, &nParts, NULL, NULL, NULL, &objval,
          part.data());

      for (int i = 0; i < part.size(); i++) {
        std::cout << part[i] << "\t" << vweight[i] << std::endl;
      }

      std::cout << ret << std::endl;
    } else {
      //      part = cholmod_solver->inspector().getNodeToSuperNode();
      //      auto perm_inv = cholmod_solver->inspector().getLastPermInv();
      //      int supernode_size_cnt = 0;
      //      for (int i = 0; i < part.size(); i++) {
      //        if (i > 0.95 * part.size()) {
      //          part[perm_inv[i]] = 1;
      //          supernode_size_cnt++;
      //        } else {
      //          part[perm_inv[i]] = 0;
      //        }
      //      }
      //    }

      part = parth_solver->inspector().getNodeToSuperNode();
      auto perm_inv = parth_solver->inspector().getLastPermInv();
      int big_supernode_id =
          part[perm_inv[parth_solver->inspector().number_of_nodes - 1]];
      for (auto &p : part) {
        if (p == big_supernode_id) {
          p = 1;
        } else {
          p = 0;
        }
      }
    }

    if (args.mat_type == "mesh") {
      std::cout << "The number of elements in each part:" << std::endl;
      double part1 = 0;
      double part2 = 0;
      for (int i = 0; i < part.size(); i++) {
        if (part[i] == 0) {
          part1++;
        } else if (part[i] == 1) {
          part2++;
        } else {
          std::cerr << "There are more than two parts" << std::endl;
        }
      }
      std::cout << "The ratio of elements in part1 is: " << part1 / part.size()
                << std::endl;
      std::cout << "The ratio of elements in part2 is: " << part2 / part.size()
                << std::endl;
      vis_obj.addIterSol(V, part.data(), std::get<0>(iter), std::get<1>(iter));
    } else {
      int DIM = 3;
      std::vector<int> reduced_part(part.size() / DIM);
      if (reduced_part.size() != V.rows()) {
        std::cout << "The hessian matrix is not for this mesh" << std::endl;
      }
      for (int i = 0; i < reduced_part.size(); i++) {
        int max = 0;
        if (part[i * 3] > max) {
          max = part[i * 3];
        }
        if (part[i * 3 + 1] > max) {
          max = part[i * 3 + 1];
        }
        if (part[i * 3 + 2] > max) {
          max = part[i * 3 + 2];
        }
        reduced_part[i] = max;
      }
      vis_obj.addIterSol(V, reduced_part.data(), std::get<0>(iter),
                         std::get<1>(iter));
      std::cout << "The number of elements in each part:" << std::endl;
      double part1 = 0;
      double part2 = 0;
      for (int i = 0; i < reduced_part.size(); i++) {
        if (reduced_part[i] == 0) {
          part1++;
        } else if (reduced_part[i] == 1) {
          part2++;
        } else {
          std::cerr << "There are more than two parts" << std::endl;
        }
      }
      std::cout << "The ratio of elements in part1 is: "
                << part1 / reduced_part.size() << std::endl;
      std::cout << "The ratio of elements in part2 is: "
                << part2 / reduced_part.size() << std::endl;
    }
  }

  vis_obj.visualizeNewtonIters(V_rest, SF, 2);
  delete parth_solver;

  return 0;
}
