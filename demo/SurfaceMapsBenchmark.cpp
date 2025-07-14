//
// Created by behrooz on 09/05/24.
//
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
#include <string>

int DIM = 4;

namespace fs = std::filesystem;

struct CLIArgs {
  std::string input;
  std::string output;
  std::string simName = "test";

  std::string solver_type = "CHOLMOD";
  std::string patch = "UPSAMPLE";
  int activate_parth = 0;
  int numThreads = 10;
  double scale = 1;

  CLIArgs(int argc, char *argv[]) {
    CLI::App app{"Patch demo"};

    app.add_option("-o,--output", output, "output folder name");
    app.add_option("-i,--input", input, "input mesh name");
    app.add_option("--scale", scale, "input mesh name");
    app.add_option("--SimName", simName, "Simulation name");
    app.add_option("--Parth", activate_parth, "ADD PARTH IMMOBILIZER");
    app.add_option("--Patch", patch, "ADD PARTH IMMOBILIZER");
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
  solver->setParthActivation(activate_parth);
  solver->parth.activate_aggressive_reuse = true;

  Eigen::SparseMatrix<double> mesh_prev;

  auto list_of_files = PARTH::getFileNames(args.input);
  std::vector<std::pair<int, int>> list_of_iter;
  for (auto &x : list_of_files) {
    auto s = PARTH::split_string(x, "_");
    if (s[0] == "map" || s[1] == "reg" || s[0] == "mesh") {
      continue;
    }
    s[2].erase(s[2].find(".mtx"), 4);
    list_of_iter.emplace_back(0, std::stoi(s[2]));
  }

  // Sort the list of iterations
  std::sort(list_of_iter.begin(), list_of_iter.end(), PARTH::sortbyFirstAndSec);
  list_of_iter.erase(std::unique(list_of_iter.begin(), list_of_iter.end()),
                     list_of_iter.end());
  Eigen::SparseMatrix<double> lower_A_csc;
  Eigen::SparseMatrix<double, Eigen::RowMajor> lower_A_csc_rowMajor;

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
        args.input + "H_reduced_" + std::to_string(iter.second) + ".mtx";
    if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
      std::cerr << "File " << hessian_name << " is not found" << std::endl;
      continue;
    }

    Eigen::SparseMatrix<double> mesh_csc =
        PARTH::computeMeshFromHessian(lower_A_csc, DIM);
    if(mesh_csc.rows() == 0){
      continue;
    }

    std::cout << "Computing Hessian for a matrix with rows: "
              << lower_A_csc.rows()
              << " and the mesh of size: " << mesh_csc.rows()
              << " with dim = " << lower_A_csc.rows() / mesh_csc.rows()
              << std::endl;
    if(lower_A_csc.rows() < 5000){
      continue;
    }

    std::string map_address = args.input + "map_" + std::to_string(iter.second) + ".mtx";
    Eigen::VectorXd map;
    if (!Eigen::loadMarketVector(map, map_address)) {
      std::cerr << "File " << map_address << " is not found" << std::endl;
    }

    //Compute new_to_old_map
    std::vector<int> new_to_old_map(mesh_csc.rows());
    int map_cnt = 0;
    for(map_cnt = 0; map_cnt < std::min(map.rows(), mesh_csc.rows()); map_cnt++){
        if(map(map_cnt) >= mesh_prev.rows()){
          break;
        }
        new_to_old_map[map_cnt] = map(map_cnt);
    }
    for(int i = map_cnt; i < mesh_csc.rows(); i++){
        new_to_old_map[i] = -1;
    }
    //***********************************************************************
    Eigen::VectorXd Parth_sol(lower_A_csc.rows());
    Parth_sol.setZero();


    if (solver->type() == PARTH_SOLVER::LinSysSolverType::MKL_LIB) {
      // Convert S from column major to row major
      lower_A_csc_rowMajor = lower_A_csc;
      // Get the Upper triangle part of SRowMajor
      lower_A_csc_rowMajor = lower_A_csc_rowMajor.triangularView<Eigen::Upper>();
#ifndef NDEBUG
      int *Sp = lower_A_csc_rowMajor.outerIndexPtr();
      int *Si = lower_A_csc_rowMajor.innerIndexPtr();
      bool has_upper = false;
      bool has_lower = false;
      for (int i = lower_A_csc_rowMajor.rows() / 2; i < lower_A_csc_rowMajor.rows(); i++) {
        for (int j = Sp[i]; j < Sp[i + 1]; j++) {
          if (i < Si[j]) {
            has_upper = true;
          } else if (i > Si[j]) {
            has_lower = true;
          }
        }
      }
      if (has_upper && has_lower) {
        std::cerr << "Symmetric" << std::endl;
      } else if (has_upper) {
        std::cout << "Upper" << std::endl;
      } else if (has_lower) {
        std::cerr << "Lower" << std::endl;
      }
#endif
      solver->setMatrix(lower_A_csc_rowMajor.outerIndexPtr(), lower_A_csc_rowMajor.innerIndexPtr(),
                        lower_A_csc_rowMajor.valuePtr(), lower_A_csc_rowMajor.rows(),
                        lower_A_csc_rowMajor.nonZeros(), mesh_csc.outerIndexPtr(),
                        mesh_csc.innerIndexPtr(), mesh_csc.rows(),
                        new_to_old_map);
    } else {
      solver->setFullCoeffMatrix(lower_A_csc);
      solver->setMatrix(lower_A_csc.outerIndexPtr(), lower_A_csc.innerIndexPtr(),
                        lower_A_csc.valuePtr(), lower_A_csc.rows(),
                        lower_A_csc.nonZeros(), mesh_csc.outerIndexPtr(),
                        mesh_csc.innerIndexPtr(), mesh_csc.rows(),
                        new_to_old_map);
    }


    // Generate a rhs with random number
    Eigen::VectorXd rhs = Eigen::VectorXd::Random(lower_A_csc.rows());

    std::cout << "++++++++++++++++++ Analysing *********************"
              << std::endl;
    if (!equalMesh(mesh_csc, mesh_prev)) {
      solver->analyze_pattern();
    }
//        MKL_Immobilizer.analyze_pattern();

    std::cout << "++++++++++++++++++ Factorize *********************"
              << std::endl;
    solver->factorize();
    std::cout << "++++++++++++++++++ Solve *********************" << std::endl;

    rhs = -rhs;
    Eigen::VectorXd sol(rhs.rows());
    solver->solve(rhs, sol);
    end_to_end_time = omp_get_wtime() - end_to_end_time;

    // Solve Quality

//      solver->total_analyze_time = solver->total_analyze_time - solver->analyze_time;
//      solver->analyze_time = 0;
//      solver->parth_time = 0;
//    }
    solver->residual =
        (rhs - lower_A_csc.selfadjointView<Eigen::Lower>() * sol).norm();
    solver->printTiming();
    solver->addCSVRecord(args.simName, 0, iter.second, end_to_end_time);

    mesh_prev = mesh_csc;
  }
  return 0;
}
