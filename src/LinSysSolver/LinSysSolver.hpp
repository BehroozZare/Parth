//
//  LinSysSolver.hpp
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once

#include "Parth.h"
#include <omp.h>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <csv_utils.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>

namespace PARTH_SOLVER {

enum class LinSysSolverType {
  CHOLMOD,
  ACCELERATE,
  MKL_LIB,
  BARB,
  LAZY_BARB,
  JACOBI_BARB,
  CG,
  PARALLEL_CHOLMOD,
  PARALLEL_LAZY_CHOLMOD,
  SYMPILER,
  EIGEN,
  PURE_METIS
};

enum class ReorderingType {
  METIS,
  AMD,
};

class LinSysSolver {
public:
  double load_time = 0;
  double analyze_time = 0;
  double factor_time = 0;
  double solve_time = 0;

  double total_load_time = 0;
  double total_factor_time = 0;
  double total_solve_time = 0;
  double total_analyze_time = 0;

  PARTH::Parth parth;
  bool activate_parth = true;
  int dim = 1;
  int lag_cnt = 0;
  double parth_time = 0;
  ReorderingType reorderingType = ReorderingType::METIS;
  std::vector<int> perm;

  int L_NNZ = 0;
  int NNZ = 0;
  int N = 0;
  double residual = 0;

  Eigen::SparseMatrix<double> mtr;

  double numerical_reuse = 0;
  double max_possible_numerical_reuse = 0;
  int num_cores = 10;

  int compression_mode = 0;//TODO:DELETE FOR RELEASED CODE

public:
  virtual ~LinSysSolver(void){};

  static LinSysSolver *create(const LinSysSolverType type);

  virtual LinSysSolverType type() const = 0;

public:
  int frame;
  int iteration;
  std::string outputFolderPath;

  virtual void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp,
                         int *Mi, int M_N,
                         std::vector<int> &new_to_old_map) = 0;

    virtual void analyze_pattern(bool no_perm = false) {
        analyze_time = omp_get_wtime();
        if (activate_parth) {
          if (reorderingType == ReorderingType::METIS) {
            parth.setReorderingType(PARTH::ReorderingType::METIS);
            std::cout << "Using METIS" << std::endl;
          } else if (reorderingType == ReorderingType::AMD) {
            parth.setReorderingType(PARTH::ReorderingType::AMD);
            std::cout << "Using AMD" << std::endl;
          }

          if (!no_perm || perm.size() != N) {
            parth_time = omp_get_wtime();
            parth.computePermutation(perm, dim);
            parth_time = omp_get_wtime() - parth_time;
          }
          assert(perm.size() == N);
        }

        innerAnalyze_pattern(no_perm);
        analyze_time = omp_get_wtime() - analyze_time;
    }

  virtual void innerAnalyze_pattern(bool no_perm) = 0;

    virtual bool factorize(void){
        factor_time = omp_get_wtime();
        bool success = innerFactorize();
        factor_time = omp_get_wtime() - factor_time;
        return success;
    }

  virtual bool innerFactorize(void) = 0;

  virtual void solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result){
      solve_time = omp_get_wtime();
      innerSolve(rhs, result);
      solve_time = omp_get_wtime() - solve_time;
      computeResidual(result, rhs);
      std::cout << "THE RESIDUAL: " << residual << std::endl;
      addTotalTime();
  }

  virtual void solve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result){
      solve_time = omp_get_wtime();
      innerSolve(rhs, result);
      solve_time = omp_get_wtime() - solve_time;
      addTotalTime();
  };

  virtual void computeResidual(Eigen::VectorXd& sol, Eigen::VectorXd& rhs){
    residual = (rhs - mtr.selfadjointView<Eigen::Lower>() * sol).norm();
  }

  virtual void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) = 0;

  virtual void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) = 0;

  virtual void resetSolver() = 0;

  virtual void setFullCoeffMatrix(Eigen::SparseMatrix<double> &A){
    mtr = A;
  }

public:
  int getNumRows(void) const { return N; }

  double getResidual(void) { return residual; }

  virtual void setReorderingType(ReorderingType reorderingType) {
    this->reorderingType = reorderingType;
  }

  virtual void setParthActivation(bool activate) {
    activate_parth = activate;
    if (activate_parth) {
      std::cout << "Activating Immobilizer" << std::endl;
    } else {
      std::cout << "Deactivating Immobilizer" << std::endl;
    }
  }

  virtual void setSimulationDIM(int dim) {
      this-> dim = dim;
      this-> parth.sim_dim = dim;
  }

  void printTiming() {
    parth.printTiming();
    double total_time = load_time + analyze_time + factor_time + solve_time;
    std::cout << "+++ PARTH: The analysis time is: " << analyze_time << " - "
              << (analyze_time / total_time) * 100 << "%" << std::endl;
    std::cout << "+++ PARTH: The factorization time is: " << factor_time
              << " - " << (factor_time / total_time) * 100 << "%" << std::endl;
    std::cout << "+++ PARTH: The solve time is: " << solve_time << " - "
              << (solve_time / total_time) * 100 << "%" << std::endl;
    std::cout << "+++ PARTH: Total Solve time: " << total_time << " seconds"
              << std::endl;
  }

  virtual void initVariables() {
    load_time = 0;
    analyze_time = 0;
    factor_time = 0;
    solve_time = 0;

    total_load_time = 0;
    total_factor_time = 0;
    total_solve_time = 0;
    total_analyze_time = 0;

    activate_parth = false;
    dim = 1;
    parth_time = 0;
    reorderingType = ReorderingType::METIS;
    perm.clear();

    L_NNZ = 0;
    NNZ = 0;
    N = 0;
    residual = 0;
  }

  // PARTH STATISTICS
  virtual double getContactSize() { return parth.getNumChanges(); }

  virtual double getReusePercentage() { return parth.getReuse(); }

  virtual double getIntegrationTime() {
    return parth.dof_change_integrator_time;
  }

  virtual double getChangeDetTime() { return parth.change_computation_time; }

  virtual double getPermCompTime() { return parth.compute_permutation_time; }

  virtual double getMapTime() {
    return parth.map_mesh_to_matrix_computation_time;
  }

  virtual double getTotalTime() {
    double parth_total_time = parth.dof_change_integrator_time +
                              parth.change_computation_time +
                              parth.compute_permutation_time +
                              parth.map_mesh_to_matrix_computation_time;
    return parth_total_time;
  }

  virtual void addCSVRecord(std::string csv_address, int frame, int iter,
                            double end_to_end_time) {
    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("N");
    Runtime_headers.emplace_back("NNZ");
    Runtime_headers.emplace_back("L_NNZ");

    Runtime_headers.emplace_back("nthreads");

    // Iteration ids
    Runtime_headers.emplace_back("FrameNum");
    Runtime_headers.emplace_back("NewtonIter");

    // Solve Quality
    Runtime_headers.emplace_back("residual");

    // Timing Headers
    Runtime_headers.emplace_back("load_time");
    Runtime_headers.emplace_back("analyze_time");
    Runtime_headers.emplace_back("parth_time");
    Runtime_headers.emplace_back("factor_time");
    Runtime_headers.emplace_back("solve_time");

    // Parth Statistic
    Runtime_headers.emplace_back("contact_size");
    Runtime_headers.emplace_back("parth_reuse");
    Runtime_headers.emplace_back("barb_reuse");
    Runtime_headers.emplace_back("max_possible_reuse");
    Runtime_headers.emplace_back("parth_integration_time");
    Runtime_headers.emplace_back("parth_change_det_time");
    Runtime_headers.emplace_back("parth_perm_comp_time");
    Runtime_headers.emplace_back("parth_map_time");
    Runtime_headers.emplace_back("parth_total_time_time");
    Runtime_headers.emplace_back("parth_reordering");

    Runtime_headers.emplace_back("total_load_time");
    Runtime_headers.emplace_back("total_analyze_time");
    Runtime_headers.emplace_back("total_factor_time");
    Runtime_headers.emplace_back("total_solve_time");

    Runtime_headers.emplace_back("end_to_end_time");

    std::string Data_name;
    PARTH::CSVManager runtime_csv(csv_address, "some address", Runtime_headers,
                                  false);

    runtime_csv.addElementToRecord(N, "N");
    runtime_csv.addElementToRecord(NNZ, "NNZ");
    runtime_csv.addElementToRecord(L_NNZ, "L_NNZ");
    runtime_csv.addElementToRecord(parth.getNumberOfCores(), "nthreads");

    // Iteration ids
    runtime_csv.addElementToRecord(frame, "FrameNum");
    runtime_csv.addElementToRecord(iter, "NewtonIter");

    runtime_csv.addElementToRecord(residual, "residual");

    // Timing Headers
    runtime_csv.addElementToRecord(load_time, "load_time");
    runtime_csv.addElementToRecord(analyze_time, "analyze_time");
    runtime_csv.addElementToRecord(parth_time, "parth_time");
    runtime_csv.addElementToRecord(factor_time, "factor_time");
    runtime_csv.addElementToRecord(solve_time, "solve_time");

    // 3 Region STATISICS
    runtime_csv.addElementToRecord(getContactSize(), "contact_size");
    runtime_csv.addElementToRecord(getReusePercentage(), "parth_reuse");
    runtime_csv.addElementToRecord(numerical_reuse, "barb_reuse");
    runtime_csv.addElementToRecord(max_possible_numerical_reuse, "max_possible_reuse");
    runtime_csv.addElementToRecord(getIntegrationTime(),
                                   "parth_integration_time");
    runtime_csv.addElementToRecord(getChangeDetTime(), "parth_change_det_time");
    runtime_csv.addElementToRecord(getPermCompTime(), "parth_perm_comp_time");
    runtime_csv.addElementToRecord(getMapTime(), "parth_map_time");
    runtime_csv.addElementToRecord(getTotalTime(), "parth_total_time_time");
    if (reorderingType == ReorderingType::METIS)
      runtime_csv.addElementToRecord("metis", "parth_reordering");
    else if (reorderingType == ReorderingType::AMD)
      runtime_csv.addElementToRecord("amd", "parth_reordering");

    runtime_csv.addElementToRecord(total_load_time, "total_load_time");
    runtime_csv.addElementToRecord(total_analyze_time, "total_analyze_time");
    runtime_csv.addElementToRecord(total_factor_time, "total_factor_time");
    runtime_csv.addElementToRecord(total_solve_time, "total_solve_time");
    runtime_csv.addElementToRecord(end_to_end_time, "end_to_end_time");

    runtime_csv.addRecord();
    resetTimer();
    parth.resetTimers();
    numerical_reuse = 0;
    max_possible_numerical_reuse = 0;
    parth.integrator.reuse_ratio = 1;
    parth.changed_dof_edges.clear();
  }

  void resetTimer() {
    analyze_time = 0;
    factor_time = 0;
    solve_time = 0;
    load_time = 0;
    parth_time = 0;
  }

  void addTotalTime() {
    total_load_time += load_time;
    total_analyze_time += analyze_time;
    total_factor_time += factor_time;
    total_solve_time += solve_time;
  }

  void resetTotalTime() {
    total_load_time = 0;
    total_analyze_time = 0;
    total_factor_time = 0;
    total_solve_time = 0;
  }

  void resetParth() {
    resetTimer();
    parth.clearParth();
  }

  void setNumberOfCores(int cores) {
      num_cores = cores;
  }
};

} // namespace PARTH_SOLVER
