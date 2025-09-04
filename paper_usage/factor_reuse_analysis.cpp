//
// Created by behrooz on 22/11/22.
//
//
// Created by behrooz on 11/10/22.
//

#include "NewtonIterVisualization.h"
#include "Parth_utils.h"
#include <CLI/CLI.hpp>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

#include "Barb.h"
#include "cholmod.h"
#define MAX_FRAME 10
#define MIN_FRAME 0
#define MAX_ITER 15

namespace fs = std::filesystem;

struct CLIArgs {
  std::string input_address = "../input";
  std::string output_address = "../output";
  std::string ordMethod = "METIS";
  std::string analyze_type = "MESH";
  std::string numeric_reuse_type = "BASE";
  int num_parts = 4;

  bool synthesize_mat = true;
  int synthesize_mat_size = 50;
  CLIArgs(int argc, char *argv[]) {
    CLI::App app{"Parth Solver"};

    // Custom flags
    app.add_option("--input", input_address, "Input address");
    app.add_option("--output", output_address, "Output address");
    app.add_option("--NumParts", num_parts, "Output address");
    app.add_option("--NumericReuse", numeric_reuse_type, "Output address");
    app.add_option("--AnalyzeType", analyze_type, "Output address");
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

  auto list_of_files = PARTH::getFileNames(args.input_address);
  std::vector<std::tuple<int, int>> list_of_iter;
  for (auto &x : list_of_files) {
    auto s = PARTH::split_string(x, "_");
    if (s.empty() || s[0] != "V" || s[1] == "rest") {
      continue;
    }
    list_of_iter.emplace_back(
        std::tuple<int, int>(std::stoi(s[1]), std::stoi(s[2])));
  }
  // Sort the list of iterations
  std::sort(list_of_iter.begin(), list_of_iter.end(), PARTH::sortbyFirstAndSec);
  list_of_iter.erase(std::unique(list_of_iter.begin(), list_of_iter.end()),
                     list_of_iter.end());

  PARTH::ParthSolver ParthSolver;
  ParthSolver.setInputAddress(args.input_address);
  ParthSolver.Options().setComputeResidual(true);
  ParthSolver.Options().setVerbose(true);
  if (args.analyze_type == "HESSIAN") {
    ParthSolver.Options().setAnalyzeType(
        PARTH::ParthSolver::AnalyzeType::Hessian);
  } else if (args.analyze_type == "MESH") {
    ParthSolver.Options().setAnalyzeType(PARTH::ParthSolver::AnalyzeType::Mesh);
  } else if (args.analyze_type == "SIM") {
    ParthSolver.Options().setAnalyzeType(
        PARTH::ParthSolver::AnalyzeType::SimulationAware);
  } else {
    std::cerr << "Unknown analysis type" << std::endl;
  }

  if (args.numeric_reuse_type == "BASE") {
    ParthSolver.Options().setNumericReuseType(
        PARTH::ParthSolver::NumericReuseType::NUMERIC_REUSE_PARALLEL);
  } else {
    ParthSolver.Options().setNumericReuseType(
        PARTH::ParthSolver::NumericReuseType::NUMERIC_NO_REUSE);
  }

  ParthSolver.Options().setReorderingType(
      PARTH::ParthSolver::ReorderingType::METIS);
  ParthSolver.Options().setNumRegions(args.num_parts);
  ParthSolver.Options().setCSVExportAddress(
      "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/");
  ParthSolver.Options().setNumberOfCores(10);

  PARTH::ParthSolver CholmodSolver;
  CholmodSolver.Options().setComputeResidual(true);
  CholmodSolver.Options().setVerbose(true);
  CholmodSolver.Options().setAnalyzeType(
      PARTH::ParthSolver::AnalyzeType::Hessian);
  CholmodSolver.Options().setReorderingType(
      PARTH::ParthSolver::ReorderingType::METIS);
  CholmodSolver.Options().setNumericReuseType(
      PARTH::ParthSolver::NumericReuseType::NUMERIC_NO_REUSE);

  // Initializing the iter view class
  PARTH::NewtonIterVisualization vis_obj = PARTH::NewtonIterVisualization();
  vis_obj.init("arma", true,
               "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/");

  int cnt = 0;

  Eigen::MatrixXd V_prev;

  Eigen::MatrixXd g_prev;

  Eigen::SparseMatrix<double> A_curr;

  std::vector<Eigen::SparseMatrix<double>> A_prev_vec(6);
  std::vector<Eigen::SparseMatrix<double>> A_curr_vec(6);

  std::vector<std::string> Runtime_headers;
  Runtime_headers.emplace_back("frame");
  Runtime_headers.emplace_back("iter");
  Runtime_headers.emplace_back("parth_symbolic_time");
  Runtime_headers.emplace_back("parth_factorization_time");
  Runtime_headers.emplace_back("parth_solve_time");
  Runtime_headers.emplace_back("parth_SpMV_time");
  Runtime_headers.emplace_back("parth_solve_Kernel_time");
  Runtime_headers.emplace_back("parth_residual");
  Runtime_headers.emplace_back("parth_steps_to_solve");
  Runtime_headers.emplace_back("parth_numeric_reuse");
  Runtime_headers.emplace_back("parth_symbolic_reuse");

  Runtime_headers.emplace_back("parth_iteration_to_converge");

  Runtime_headers.emplace_back("chol_symbolic_time");
  Runtime_headers.emplace_back("chol_factorization_time");
  Runtime_headers.emplace_back("chol_solve_time");
  Runtime_headers.emplace_back("chol_residual");
  profiling_utils::CSVManager runtime_csv(
      "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/final",
      "some address", Runtime_headers, false);

  // Main loop for analyzing iterations of a frame
  for (auto &iter : list_of_iter) {
    if (iter == list_of_iter.back()) {
      break;
    }
    cnt++;
    if (std::get<0>(iter) == MAX_FRAME) {
      break;
    }
    if (std::get<0>(iter) < MIN_FRAME) {
      continue;
    }
    //    if (cnt > MAX_ITER) {
    //      break;
    //    }

    ParthSolver.regions.frame = std::get<0>(iter);
    ParthSolver.regions.iter = std::get<1>(iter);
    std::cout << "+++++++++++++++++++++++ PERFORMING FRAME: "
              << ParthSolver.regions.frame
              << " and ITER: " << ParthSolver.regions.iter << std::endl;
    ParthSolver.setFrameIter(std::get<0>(iter), std::get<1>(iter), "test");
    //************** Load the Mesh and hessian *************
    Eigen::SparseMatrix<double> mesh_csc;
    std::string mesh_name = args.input_address + "mesh_" +
                            std::to_string(std::get<0>(iter)) + "_" +
                            std::to_string(std::get<1>(iter)) + "_" + "IPC.mtx";
    if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
      std::cerr << "File " << mesh_name << " is not found" << std::endl;
    }

    Eigen::SparseMatrix<double> lower_A_csc;
    std::string hessian_name =
        args.input_address + "hessian_" + std::to_string(std::get<0>(iter)) +
        "_" + std::to_string(std::get<1>(iter)) + "_" + "last_IPC.mtx";

    if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
      std::cerr << "File " << hessian_name << " is not found" << std::endl;
    }

    Eigen::MatrixXd V_curr = PARTH::openMatDouble(
        args.input_address + "V_" + std::to_string(std::get<0>(iter)) + "_" +
        std::to_string(std::get<1>(iter)));

    Eigen::MatrixXd g_curr = PARTH::openMatDouble(
        args.input_address + "rhs_" + std::to_string(std::get<0>(iter)) + "_" +
        std::to_string(std::get<1>(iter)));

    Eigen::MatrixXd cholmod_sol = PARTH::openMatDouble(
        args.input_address + "sol_" + std::to_string(std::get<0>(iter)) + "_" +
        std::to_string(std::get<1>(iter)));

    std::cout << "sol size: " << cholmod_sol.rows() << std::endl;
    std::cout << "Mesh size: " << mesh_csc.rows() << std::endl;
    std::cout << "Matrix size: " << lower_A_csc.rows() << std::endl;

    //***********************************************************************
    Eigen::VectorXd Parth_sol(lower_A_csc.rows());
    Parth_sol.setZero();

    ParthSolver.setMatrixPointers(
        lower_A_csc.outerIndexPtr(), lower_A_csc.innerIndexPtr(),
        lower_A_csc.valuePtr(), lower_A_csc.rows(), lower_A_csc.nonZeros());
    ParthSolver.setMeshPointers(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                                mesh_csc.innerIndexPtr());

    CholmodSolver.setMatrixPointers(
        lower_A_csc.outerIndexPtr(), lower_A_csc.innerIndexPtr(),
        lower_A_csc.valuePtr(), lower_A_csc.rows(), lower_A_csc.nonZeros());
    CholmodSolver.setMeshPointers(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                                  mesh_csc.innerIndexPtr());

    Eigen::VectorXd rhs(Eigen::Map<Eigen::VectorXd>(
        g_curr.data(), g_curr.cols() * g_curr.rows()));

    double parth_symbolic_t = omp_get_wtime();
    ParthSolver.analyze();
    parth_symbolic_t = omp_get_wtime() - parth_symbolic_t;

    double parth_factor_t = omp_get_wtime();
    ParthSolver.factorize();
    parth_factor_t = omp_get_wtime() - parth_factor_t;
    double parth_solve_t = omp_get_wtime();
    ParthSolver.solve(rhs, Parth_sol);
    parth_solve_t = omp_get_wtime() - parth_solve_t;

    double cholmod_symbolic_t = 0, cholmod_factor_t = 0;

    cholmod_symbolic_t = omp_get_wtime();
    CholmodSolver.analyze();
    cholmod_symbolic_t = omp_get_wtime() - cholmod_symbolic_t;

    cholmod_factor_t = omp_get_wtime();
    CholmodSolver.factorize();
    cholmod_factor_t = omp_get_wtime() - cholmod_factor_t;

    double cholmod_solve_t = omp_get_wtime();
    CholmodSolver.solve(rhs, Parth_sol);
    cholmod_solve_t = omp_get_wtime() - cholmod_solve_t;

    runtime_csv.addElementToRecord(ParthSolver.regions.frame, "frame");
    runtime_csv.addElementToRecord(ParthSolver.regions.iter, "iter");

    parth_symbolic_t = ParthSolver.getAnalyzeTime();
    parth_factor_t = ParthSolver.getFactorTime();
    parth_solve_t = ParthSolver.getSolveTime();

    runtime_csv.addElementToRecord(parth_symbolic_t, "parth_symbolic_time");
    runtime_csv.addElementToRecord(parth_factor_t, "parth_factorization_time");
    runtime_csv.addElementToRecord(parth_solve_t, "parth_solve_time");
    runtime_csv.addElementToRecord(ParthSolver.SpMV_total_time,
                                   "parth_SpMV_time");
    runtime_csv.addElementToRecord(ParthSolver.solve_kernel_total_time,
                                   "parth_solve_Kernel_time");

    runtime_csv.addElementToRecord(ParthSolver.getResidual(), "parth_residual");
    runtime_csv.addElementToRecord(ParthSolver.getNumericReuse(),
                                   "parth_numeric_reuse");
    //    runtime_csv.addElementToRecord(0, "parth_numeric_reuse");
    runtime_csv.addElementToRecord(ParthSolver.getSymbolicReuse(),
                                   "parth_symbolic_reuse");
    runtime_csv.addElementToRecord(ParthSolver.num_steps_to_get_result,
                                   "parth_steps_to_solve");
    runtime_csv.addElementToRecord(ParthSolver.num_refinement_iteration,
                                   "parth_iteration_to_converge");

    //    runtime_csv.addElementToRecord(0, "chol_symbolic_time");
    //    runtime_csv.addElementToRecord(0, "chol_factorization_time");
    //    runtime_csv.addElementToRecord(0, "chol_solve_time");

    runtime_csv.addElementToRecord(cholmod_symbolic_t, "chol_symbolic_time");
    runtime_csv.addElementToRecord(cholmod_factor_t, "chol_factorization_time");
    runtime_csv.addElementToRecord(cholmod_solve_t, "chol_solve_time");
    runtime_csv.addElementToRecord(CholmodSolver.getResidual(),
                                   "chol_residual");
    //    runtime_csv.addElementToRecord(0, "chol_residual");
    runtime_csv.addRecord();

    std::cout << "The reuse is: " << ParthSolver.getNumericReuse() << std::endl;
    std::cout << "The residual is: " << ParthSolver.getResidual() << std::endl;

    auto hessian_perm = ParthSolver.getHessianPerm();
    auto mesh_perm = ParthSolver.getMeshPerm();
    if (V_prev.rows() == 0) {
      V_prev = V_curr;
    }

    if (g_prev.rows() == 0) {
      g_prev = g_curr;
    }
    //        std::vector<int> contacts;
    //        ParthSolver.computeContacts(contacts);
    //        if (contacts.size() != 0) {
    //          std::cerr << "Contact happened" << std::endl;
    //          break;
    //        }
    //
    //        ParthSolver.permuteMatrix(lower_A_csc.rows(),
    //        lower_A_csc.outerIndexPtr(),
    //                                  lower_A_csc.innerIndexPtr(),
    //                                  lower_A_csc.valuePtr(), hessian_perm,
    //                                  A_curr);

    Eigen::VectorXd cholmod_sol_vec(Eigen::Map<Eigen::VectorXd>(
        cholmod_sol.data(), cholmod_sol.cols() * cholmod_sol.rows()));

    V_prev = V_curr;
    g_prev = g_curr;

    assert(ParthSolver.getElemRegions().size() == V_curr.rows());
    vis_obj.addIterSol(V_curr, ParthSolver.getElemRegions().data(),
                       std::get<0>(iter), std::get<1>(iter));
    //    int cnt = 0;
    //    for (auto &cache : ParthSolver.caching_history) {
    //      double cache_cnt = 0;
    //      for (auto &iter : cache) {
    //        if (iter != 0) {
    //          cache_cnt++;
    //        }
    //      }
    //      std::cout << "The non-zeros ratio in this vector is "
    //                << cache_cnt / cache.size() << std::endl;
    //
    //      std::string name;
    //      if (cnt == 0) {
    //        name = std::to_string(std::get<0>(iter)) + "_" +
    //               std::to_string(std::get<1>(iter)) + "_reuse_" +
    //               std::to_string(cache_cnt / cache.size());
    //      }
    //
    //      if (cnt == 1) {
    //        name = std::to_string(std::get<0>(iter)) + "_" +
    //               std::to_string(std::get<1>(iter)) + "_SPD_" +
    //               std::to_string(cache_cnt / cache.size());
    //      }
    //
    //      vis_obj.addIterSol(V_curr, cache.data(), std::get<0>(iter),
    //                         std::get<1>(iter), name);
    //      cnt++;
    //    }

    //    int cnt = 0;
    //    for (auto &cache : ParthSolver.score_view) {
    //      std::string name;
    //      name = std::to_string(std::get<0>(iter)) + "_" +
    //             std::to_string(std::get<1>(iter)) + "_" +
    //             ParthSolver.score_name[cnt++];
    //      vis_obj.addIterSol(V_curr, cache.data(), std::get<0>(iter),
    //                         std::get<1>(iter), name);
    //    }
  }

  // Load the SF matrix
  Eigen::MatrixXi SF = PARTH::openMatInt(args.input_address + "SF");

  // Load the V_rest matrix
  Eigen::MatrixXd V_rest = PARTH::openMatDouble(args.input_address + "V_rest");

  vis_obj.visualizeNewtonIters(V_rest, SF, 100);

  return 0;
}
