//
// Created by behrooz on 28/09/22.
//

#include "Parth_utils.h"
#include "Solver.h"
#include "cholmod.h"
#include <CLI/CLI.hpp>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>

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
  Eigen::SparseMatrix<double> lower_A_csc;
  Eigen::SparseMatrix<double> full_A_csc;

  // Timing variables
  double chol_analysis_time, chol_factor_time, chol_solve_time;

  // Read Matrix
  if (args.synthesize_mat) {
    Eigen::SparseMatrix<double, Eigen::RowMajor> tmp;
    PARTH::compute3DPossion(tmp, args.synthesize_mat_size);
    full_A_csc = tmp;
    lower_A_csc = full_A_csc.triangularView<Eigen::Lower>();
  } else {
    Eigen::loadMarket(lower_A_csc,
                      "/home/behrooz/Desktop/IPC_Project/Parth/input/ "
                      "sphere_mat_hessian_64_4_PreOrder.mtx");
    full_A_csc = lower_A_csc.selfadjointView<Eigen::Lower>();
  }


  //  //==================== cholmod ====================
  //  auto cholmod_solver =
  //  PARTH::Solver::create(PARTH::LinSysSolverType::CHOLMOD);
  //  cholmod_solver->setMatrix(lower_A_csc);
  //  cholmod_solver->analyze();
  //  cholmod_solver->factorize();
  //
  //  cholmod_solver->solve(b_eigen, x_eigen);
  //
  //  std::cout << "Cholmod: reordering time is: "
  //            << cholmod_solver->getAnalyzeTime() << std::endl;
  //  std::cout << "Cholmod: factorization time is: "
  //            << cholmod_solver->getFactorTime() << std::endl;
  //  std::cout << "Cholmod: solve time is: " << cholmod_solver->getSolveTime()
  //            << std::endl;
  //  std::cout << "Cholmod: Res is:" << cholmod_solver->getResidual() <<
  //  std::endl; delete cholmod_solver;

  //==================== My Framework ====================
  for (int i = 0; i < 1; i++) {
    Eigen::VectorXd x_eigen(lower_A_csc.rows());
    Eigen::VectorXd b_eigen(lower_A_csc.rows());
    x_eigen.setZero();
    b_eigen.setOnes();
    std::string out_address;
    std::string sim_name;
    if (i == 0) {
      Eigen::loadMarket(
          lower_A_csc,
          "/home/behrooz/Desktop/IPC_Project/SolverTestBench/input/"
          "hessian_0_0_IPC.mtx");
      out_address = "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/"
                    "hessian_0_0_IPC";
    }

    if (i == 1) {
      Eigen::loadMarket(
          lower_A_csc,
          "/home/behrooz/Desktop/IPC_Project/SolverTestBench/input/"
          "4_rodsTwist_800_0_IPC.mtx");
      out_address = "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/"
                    "4_rodsTwist";
    }

    if (i == 2) {
      Eigen::loadMarket(
          lower_A_csc,
          "/home/behrooz/Desktop/IPC_Project/SolverTestBench/input/"
          "16_armaRoller_150_0_IPC.mtx");
      out_address = "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/"
                    "16_armaRoller_E1e5";
    }
    if (i == 3) {
      Eigen::SparseMatrix<double, Eigen::RowMajor> tmp;
      PARTH::compute3DPossion(tmp, 50);
      full_A_csc = tmp;
      lower_A_csc = full_A_csc.triangularView<Eigen::Lower>();
      out_address = "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/"
                    "3D_Poisson_50";
    }
    auto cholmod_solver =
        PARTH::Solver::create(PARTH::LinSysSolverType::CHOLMOD_Profile);
    cholmod_solver->options().setCSVOutPutAddress(out_address);

    cholmod_solver->setMatrix(lower_A_csc);
    cholmod_solver->analyze();
    cholmod_solver->factorize();
    cholmod_solver->solve(b_eigen, x_eigen);
    cholmod_solver->outputA(out_address + "_A.mtx");
    //    cholmod_solver->outputFactorization(out_address + "_L.mtx");
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

    std::cout << "The average supernode size is:"
              << cholmod_solver->inspector().getAvgSuperNodeSize() << std::endl;

    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("ID");
    Runtime_headers.emplace_back("Timing");
    profiling_utils::CSVManager runtime_csv(out_address, "some address",
                                            Runtime_headers, false);

    for (int j = 0; j < cholmod_solver->inspector().number_of_supernodes; j++) {
      runtime_csv.addElementToRecord(j, "ID");
      runtime_csv.addElementToRecord(
          cholmod_solver->inspector().supernode_computation_timing[j],
          "Timing");
      runtime_csv.addRecord();
    }

    delete cholmod_solver;
  }

  return 0;
}
