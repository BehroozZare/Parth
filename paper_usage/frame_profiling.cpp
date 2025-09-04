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
#include <unsupported/Eigen/SparseExtra>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

struct CLIArgs {
  std::string input_address = "../input";
  std::string output_address = "../output";
  std::string ordMethod = "METIS";

  bool synthesize_mat = true;
  int synthesize_mat_size = 50;
  CLIArgs(int argc, char *argv[]) {
    CLI::App app{"Parth Solver"};

    // Custom flags
    app.add_option("--input", input_address, "Input address");
    app.add_option("--output", output_address, "Output address");
    app.add_flag("--Synthesize", synthesize_mat,
                 "Flag to use 3D Poisson synthesize matrix");
    app.add_option("--SynMatSize", synthesize_mat_size,
                   "Size of the synthesize matrix");

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

  std::string path = "/media/LargeHard/IPC/test_out/"
                     "roller0-roller1-roller2_DCOVerschoorRoller_FCR_BE_"
                     "interiorPoint_20221012011848/";
  auto list_of_files = PARTH::getFileNames(path);
  std::vector<std::tuple<int, int>> list_of_iter;
  for (auto &x : list_of_files) {
    auto s = PARTH::split_string(x, "_");
    if (s.empty() || s[0] != "hessian") {
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

  auto cholmod_solver =
      PARTH::Solver::create(PARTH::LinSysSolverType::CHOLMOD_Profile);

  // Initializing the iter view class
  PARTH::NewtonIterVisualization vis_obj = PARTH::NewtonIterVisualization();
  vis_obj.init("arma", true,
               "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/");

  int max_iterations = 200;
  int cnt = 0;
  // Main loop for analyzing iterations of a frame
  for (auto &iter : list_of_iter) {
    cnt++;
    if (cnt == max_iterations) {
      break;
    }
    Eigen::SparseMatrix<double> lower_A_csc;
    std::string file_name = path + "hessian_" +
                            std::to_string(std::get<0>(iter)) + "_" +
                            std::to_string(std::get<1>(iter)) + "_" + "IPC.mtx";
    std::cout << file_name << std::endl;
    //==================== My Framework ====================
    //************** Load the hessian *************
    if (!Eigen::loadMarket(lower_A_csc, file_name)) {
      std::cerr << "File " << file_name << " is not found" << std::endl;
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

    cholmod_solver->setMatrix(lower_A_csc);
    cholmod_solver->analyze();

    std::cout << "Profile: reordering time is: "
              << cholmod_solver->getAnalyzeTime() << std::endl;

    std::cout << "Profile: Permutation vector change - Cosine: "
              << cholmod_solver->inspector().getPermutationChange()
              << std::endl;

        //    auto node_to_supernode =
        //    cholmod_solver->inspector().getNodeToSuperNode();
        auto node_to_supernode =
        cholmod_solver->inspector().getNodeToSuperNode(); auto group_id_set =
        PARTH::getBiggestIds(
            cholmod_solver->inspector().supernode_computation_timing);

        // Get supernodes that contribute to more than 30% of overall runtime
        double total_time = 0;
        for (auto &iter :
             cholmod_solver->inspector().supernode_computation_timing) {
          total_time += iter;
        }

        // Find the number of supernodes
        double sub_time = 0;
        int num_big_supernodes;
        for (num_big_supernodes = 0;
             num_big_supernodes <
             cholmod_solver->inspector().number_of_supernodes;
             num_big_supernodes++) {
          sub_time +=
              cholmod_solver->inspector()
                  .supernode_computation_timing[group_id_set[num_big_supernodes]];
          if (sub_time / total_time > 0.5) {
            break;
          }
        }

        std::cout
            << "Profile: #number of supernodes /"
               " #total number of supernodes for more than 50% of the runtime:
               "
            << num_big_supernodes * 1.0 /
                   cholmod_solver->inspector().number_of_supernodes
            << std::endl;

        std::vector<int> sub_big_supernode_ids;
        for (int i = 0; i < num_big_supernodes; i++) {
          sub_big_supernode_ids.emplace_back(group_id_set[i]);
        }

        // For debugging //TODO: clean this code
        int tmp = sub_big_supernode_ids[1];
        sub_big_supernode_ids.clear();
        sub_big_supernode_ids.emplace_back(tmp);

        int supernode_size_cnt = 0;
        for (int i = 0; i < node_to_supernode.size(); i++) {
          if ((std::find(sub_big_supernode_ids.begin(),
          sub_big_supernode_ids.end(),
                         node_to_supernode[i]) !=
                         sub_big_supernode_ids.end())) {
            node_to_supernode[i] = 1;
            supernode_size_cnt++;
          } else {
            node_to_supernode[i] = 0;
          }
        }
        std::cout << "Profile: The combined size of the biggest supernodes /
        total "
                     "size of the rows is: "
                  << supernode_size_cnt * 1.0 /
                         cholmod_solver->inspector().number_of_nodes
                  << std::endl;

        std::vector<int> group_vec(cholmod_solver->inspector().number_of_nodes
        / 3); for (int i = 0; i < group_vec.size(); i++) {
          int max = 0;
          if (node_to_supernode[i * 3] > max) {
            max = node_to_supernode[i * 3];
          }
          if (node_to_supernode[i * 3 + 1] > max) {
            max = node_to_supernode[i * 3 + 1];
          }
          if (node_to_supernode[i * 3 + 2] > max) {
            max = node_to_supernode[i * 3 + 2];
          }
          group_vec[i] = max;
        }
        vis_obj.addIterSol(V, group_vec.data(), std::get<0>(iter),
                           std::get<1>(iter));
    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("iter");
    Runtime_headers.emplace_back("cos");
    profiling_utils::CSVManager runtime_csv(
        "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/"
        "16_armaRoller_E1e5_",
        "some address", Runtime_headers, false);

    runtime_csv.addElementToRecord(cnt, "iter");
    runtime_csv.addElementToRecord(
        cholmod_solver->inspector().getPermutationChange(), "cos");
    runtime_csv.addRecord();
  }

    vis_obj.visualizeNewtonIters(V_rest, SF, 2);
  delete cholmod_solver;

  return 0;
}
