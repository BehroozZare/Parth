//
// Created by behrooz on 22/11/22.
//

#include "LinSysSolver.hpp"
#include "Parth_utils.h"

#include <CLI/CLI.hpp>
#include <Eigen/Eigen>
#include <algorithm>
#include <cholmod.h>
#include <unsupported/Eigen/SparseExtra>

#include <iostream>
#include <omp.h>
#include <string>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

int DIM = 3;

namespace fs = std::filesystem;

struct CLIArgs {
  std::string input;
  std::string output;
  std::string simName = "test";

  std::string solver_type = "CHOLMOD";
  std::string patch = "UPSAMPLE";
  int activate_parth = 0;
  int numThreads = 4;
  int lagging = -1;
  double scale = 1;
  int aggressive = 1;
  int plag = 0;
  std::string order = "METIS";

  CLIArgs(int argc, char *argv[]) {
    CLI::App app{"Patch demo"};

    app.add_option("-o,--output", output, "output folder name");
    app.add_option("-i,--input", input, "input mesh name");
    app.add_option("--SimName", simName, "Simulation name");
    app.add_option("--Parth", activate_parth, "ADD PARTH IMMOBILIZER");
    app.add_option("--PAggressive", aggressive, "ADD PARTH IMMOBILIZER");
    app.add_option("--Order", order, "ADD PARTH IMMOBILIZER");
    app.add_option("--PLAG", plag, "ADD PARTH IMMOBILIZER");

    app.add_option("--Lag", lagging, "ADD PARTH IMMOBILIZER");

    app.add_option("--numThreads", numThreads,
                   "maximum number of threads to use")
        ->default_val(numThreads);
    app.add_option("--SolverType", solver_type,
                   "Choose one of the solvers CHOLMOD, EIGEN, MKL, STRUMPACK");
    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      exit(app.exit(e));
    }
  }
};

bool equalMesh(Eigen::SparseMatrix<double> &mesh_curr,
               Eigen::SparseMatrix<double> &mesh_prev) {
  int *Mp_curr = mesh_curr.outerIndexPtr();
  int *Mi_curr = mesh_curr.innerIndexPtr();
  int M_n_curr = mesh_curr.rows();

  int *Mp_prev = mesh_prev.outerIndexPtr();
  int *Mi_prev = mesh_prev.innerIndexPtr();
  int M_n_prev = mesh_prev.rows();

  if (M_n_curr != M_n_prev) {
    return false;
  }

  int NNZ = Mp_curr[M_n_curr];

  for (int n = 0; n < M_n_curr + 1; n++) {
    if (Mp_curr[n] != Mp_prev[n]) {
      return false;
    }
  }

  for (int nnz = 0; nnz < NNZ; nnz++) {
    if (Mi_curr[nnz] != Mi_prev[nnz]) {
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  CLIArgs args(argc, argv);

  omp_set_num_threads(args.numThreads);

  // Solver Config
  bool activate_parth = args.activate_parth;
  // Creating the solver
  std::string data_name = args.simName;
  PARTH_SOLVER::LinSysSolver *solver;
  if (args.solver_type == "CHOLMOD") {
    solver = PARTH_SOLVER::LinSysSolver::create(
        PARTH_SOLVER::LinSysSolverType::CHOLMOD);
  } else if (args.solver_type == "ACCELERATE") {
    solver = PARTH_SOLVER::LinSysSolver::create(
        PARTH_SOLVER::LinSysSolverType::ACCELERATE);
  } else if (args.solver_type == "MKL") {
    solver =
        PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::MKL_LIB);
  } else if (args.solver_type == "BARB") {
    solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::BARB);
  } else if (args.solver_type == "LAZY_BARB") {
    solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::LAZY_BARB);
  } else if (args.solver_type == "JACOBI_BARB") {
    solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::JACOBI_BARB);
  } else if (args.solver_type == "CG") {
    solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::CG);
  } else if (args.solver_type == "PARALLEL_CHOLMOD") {
    solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::PARALLEL_CHOLMOD);
  } else {
    std::cerr << "UNKNOWN linear solver" << std::endl;
    return 0;
  }

  if (solver == nullptr) {
    std::cerr << "Solver is not created" << std::endl;
    return 0;
  }
  // Solver Config
  solver->setSimulationDIM(DIM);
  if (args.solver_type == "BARB") {
    solver->setParthActivation(true);
  } else {
    solver->setParthActivation(activate_parth);
  }
  solver->parth.activate_aggressive_reuse = args.aggressive;
  solver->parth.lagging = args.plag;
  if (args.order == "METIS") {
    solver->reorderingType = PARTH_SOLVER::ReorderingType::METIS;
  } else if (args.order == "AMD") {
    solver->reorderingType = PARTH_SOLVER::ReorderingType::AMD;
  } else {
    std::cerr << "UNKNOWN reordering type" << std::endl;
    return 0;
  }

  Eigen::SparseMatrix<double> mesh_prev;

  auto list_of_files = PARTH::getFileNames(args.input);
  std::vector<std::pair<int, int>> list_of_iter;

  for (auto &x : list_of_files) {
    auto s = PARTH::split_string(x, "_");
    if (s.empty() || s[0] != "hessian" || s[1] == "map") {
      continue;
    }
    list_of_iter.emplace_back(std::stoi(s[1]), std::stoi(s[2]));
  }

  // Sort the list of iterations
  std::sort(list_of_iter.begin(), list_of_iter.end(), PARTH::sortbyFirstAndSec);
  list_of_iter.erase(std::unique(list_of_iter.begin(), list_of_iter.end()),
                     list_of_iter.end());
  Eigen::SparseMatrix<double> lower_A_csc;
  Eigen::SparseMatrix<double, Eigen::RowMajor> lower_A_csc_rowMajor;

  int saved_frame;

  // Main loop for analyzing iterations of a frame
  for (auto &iter : list_of_iter) {
    if (iter == list_of_iter.back()) {
      break;
    }

    double end_to_end_time = 0;

    int frame = iter.first;
    int iteration = iter.second;

    std::cout << "------------------- Testing for Frame " << frame
              << " and Iteration " << iteration << " -------------------"
              << std::endl;

    std::string hessian_name =
        args.input + "hessian_" + std::to_string(std::get<0>(iter)) + "_" +
        std::to_string(std::get<1>(iter)) + "_" + "ArcSim.mtx";

    if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
      std::cerr << "File " << hessian_name << " is not found" << std::endl;
      continue;
    }

    Eigen::SparseMatrix<double> mesh_csc =
        PARTH::computeMeshFromHessian(lower_A_csc, DIM);
    if (mesh_csc.rows() == 0) {
      continue;
    }

    std::string map_address = args.input + "map_" +
                              std::to_string(std::get<0>(iter)) + "_" +
                              std::to_string(std::get<1>(iter));

    std::ifstream in(map_address.c_str(), std::ios::in);
    // First, determine the number of lines in the file to size the vector
    int n = 0;
    std::string line;
    std::vector<int> map;
    // Read the file again and populate the vector

    while (std::getline(in, line)) {
      if (!line.empty()) {
        map.push_back(
            std::stoi(line)); // Convert line to double and store in vector
      }
    }

    if (frame == 152) {
      std::cout << "Map size: " << map.size() << std::endl;
    }
    if(map.size() != mesh_csc.rows() && map.size() != 0){
      map.clear();
      std::cerr << "The mapping has problem." << std::endl;
    }

    std::vector<bool> repeat_flag(solver->parth.M_n_prev, false);
    std::vector<int> new_to_old_map;
    if (solver->parth.M_n_prev != 0) {
      if (map.size() == 0) {
        new_to_old_map.clear();
      } else {
        for (int i = 0; i < map.size(); i++) {
          if (map[i] < solver->parth.M_n_prev) {
            if(repeat_flag[map[i]]){
              new_to_old_map.push_back(-1);
            } else {
              new_to_old_map.push_back(map[i]);
              repeat_flag[map[i]] = true;
            }
          } else {
            new_to_old_map.push_back(-1);
          }
        }
      }
    }
    assert(new_to_old_map.size() == mesh_csc.rows() ||
           new_to_old_map.size() == 0);
    std::cout << "Computing Hessian for a matrix with rows: "
              << lower_A_csc.rows()
              << " and the mesh of size: " << mesh_csc.rows()
              << " with dim = " << lower_A_csc.rows() / mesh_csc.rows()
              << std::endl;

    //***********************************************************************
    Eigen::VectorXd Parth_sol(lower_A_csc.rows());
    Parth_sol.setZero();

#ifndef NDEBUG
    int *Sp = lower_A_csc.outerIndexPtr();
    int *Si = lower_A_csc.innerIndexPtr();
    bool has_upper = false;
    bool has_lower = false;
    for (int i = 0; i < lower_A_csc.rows(); i++) {
      for (int j = Sp[i]; j < Sp[i + 1]; j++) {
        if (i < Si[j]) {
          has_lower = true;
        } else if (i > Si[j]) {
          has_upper = true;
        }
      }
    }
    if (has_upper && has_lower) {
      std::cout << "Symmetric" << std::endl;
    } else if (has_upper) {
      std::cout << "Upper" << std::endl;
    } else if (has_lower) {
      std::cout << "Lower" << std::endl;
    }
#endif
    Eigen::SparseMatrix<double> full_A =
        lower_A_csc.selfadjointView<Eigen::Lower>();
    solver->setFullCoeffMatrix(full_A);

    solver->setMatrix(lower_A_csc.outerIndexPtr(), lower_A_csc.innerIndexPtr(),
                      lower_A_csc.valuePtr(), lower_A_csc.rows(),
                      lower_A_csc.nonZeros(), mesh_csc.outerIndexPtr(),
                      mesh_csc.innerIndexPtr(), mesh_csc.rows(),
                      new_to_old_map);

    // Generate a rhs with random number
    Eigen::VectorXd rhs = Eigen::VectorXd::Random(lower_A_csc.rows());

    std::cout << "++++++++++++++++++ Analysing *********************"
              << std::endl;
    if (!equalMesh(mesh_csc, mesh_prev)) {
      bool no_perm = true;
      if (frame % args.lagging == 0 && saved_frame != frame) {
        no_perm = false;
        saved_frame = frame;
      }
      if (args.lagging == -1) {
        no_perm = false;
      }
      if (no_perm) {
        std::cout << "lag the permutation computation" << std::endl;
      }
      solver->analyze_pattern(no_perm);
    }

    std::cout << "++++++++++++++++++ Factorize *********************"
              << std::endl;
    solver->factorize();
    std::cout << "++++++++++++++++++ Solve *********************" << std::endl;
    Eigen::VectorXd sol(rhs.rows());
    solver->solve(rhs, sol);
    end_to_end_time = omp_get_wtime() - end_to_end_time;

    solver->printTiming();
    solver->addCSVRecord(args.simName, iter.first, iter.second,
                         end_to_end_time);

    mesh_prev = mesh_csc;

    //Read the obj file
    std::string obj_base = "/media/behrooz/FarazHard/ArcSim/PaperTearing/OBJs/";
    std::string texture_base = "/media/behrooz/FarazHard/ArcSim/PaperTearing/TeaserObjs/";
    std::string obj_address = obj_base + "00" + std::to_string(frame) + "_00.obj";
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(obj_address, V, F);

    //Add the appropriate texture to the obj file
    int total_num_groups = solver->parth.hmd.num_HMD_nodes;
    int lvl = solver->parth.hmd.num_levels;
    std::vector<int> dirty_nodes = solver->parth.integrator.getDirtyNodes(solver->parth.hmd);
    solver->parth.integrator.dirty_fine_HMD_nodes_saved.clear();
    solver->parth.integrator.dirty_coarse_HMD_nodes_saved.clear();
    std::vector<int> group(solver->parth.M_n, 0);
    if(!dirty_nodes.empty()){
      std::cerr << "dirty nodes in: " << frame << " " << iteration << std::endl;
    }
    for(auto& node: dirty_nodes){
      group[node] = -1;
    }

    PARTH::applyTextureBasedOnLeavesAndSeparator(V, F, lvl,
                                                 group,
                                                 texture_base, frame, iteration);

    //Save the obj file
    std::string output_obj = texture_base + "obj_" + std::to_string(frame) + "_" +
                                 std::to_string(iteration) + ".obj";
    igl::writeOBJ(output_obj, V, F);
  }
  delete solver;
  return 0;
}
