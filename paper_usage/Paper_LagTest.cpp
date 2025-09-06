
#include "LinSysSolver.hpp"
#include "Parth_utils.h"
#include <CLI/CLI.hpp>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <igl/cotmatrix.h>
#include <igl/avg_edge_length.h>

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

int main(int argc, char *argv[]) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::read_triangle_mesh(argv[1], V, F);

        //Init solver
        PARTH_SOLVER::LinSysSolver *solver;
        solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::MKL_LIB);


        if (solver == nullptr) {
                std::cerr << "Solver is not created" << std::endl;
                return 0;
        }

        // Solver Config
        solver->setSimulationDIM(1);
        solver->parth.activate_aggressive_reuse = false;
        solver->parth.lagging = false;
        if (args.order == "METIS") {
                solver->reorderingType = PARTH_SOLVER::ReorderingType::METIS;
        } else if (args.order == "AMD") {
                solver->reorderingType = PARTH_SOLVER::ReorderingType::AMD;
        } else {
                std::cerr << "UNKNOWN reordering type" << std::endl;
                return 0;
        }














        // Find boundary edges
        Eigen::MatrixXi E;
        igl::boundary_facets(F, E);
        // Find boundary vertices
        Eigen::VectorXi b, IA, IC;
        igl::unique(E, b, IA, IC);
        // List of all vertex indices
        Eigen::VectorXi all, in;
        igl::colon<int>(0, V.rows() - 1, all);
        // List of interior indices
        igl::setdiff(all, b, in, IA);

        // Construct and slice up Laplacian
        Eigen::SparseMatrix<double> L, L_in_in, L_in_b;
        igl::cotmatrix(V, F, L);
        igl::slice(L, in, in, L_in_in);
        igl::slice(L, in, b, L_in_b);

        // Dirichlet boundary conditions from z-coordinate
        Eigen::VectorXd Z = V.col(2);
        Eigen::VectorXd bc = Z(b);

        // Solve PDE
        Eigen::SparseMatrix<double> mesh_csc =
                PARTH::computeMeshFromHessian(L_in_in, 1);

        PARTH_SOLVER::LinSysSolver* solver;
        if(args.solver_type == "CHOLMOD"){
                solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::CHOLMOD);
                parth_data_name = args.output + args.simName + "_CHOLMOD_PARTH";
                basic_data_name = args.output + args.simName + "_CHOLMOD";
        } else if(args.solver_type == "ACCELERATE"){
                solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::ACCELERATE);
                parth_data_name = args.output + args.simName + "_ACCELERATE_PARTH";
                basic_data_name = args.output + args.simName + "_ACCELERATE";
        } else if(args.solver_type == "MKL"){
                solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::MKL);
                parth_data_name = args.output + args.simName + "_MKL_PARTH";
                basic_data_name = args.output + args.simName + "_MKL";
        } else {
                std::cerr << "UNKNOWN linear solver" << std::endl;
                return 0;
        }



        solver->setMatrix(
        Minus_L_in_in.outerIndexPtr(), Minus_L_in_in.innerIndexPtr(),
        Minus_L_in_in.valuePtr(), Minus_L_in_in.rows(), Minus_L_in_in.nonZeros(),
        mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(), mesh_csc.rows(),
        parth_new_to_old_map);

        //Analyze
        solver->analyze_pattern();

        //Factorize
        solver->factorize();

        //Solve
        Eigen::VectorXd rhs = L_in_b * bc;
        Eigen::VectorXd sol;
        solver->solve(rhs, sol);
        Z(in) = sol;
        end_to_end_time = omp_get_wtime() - end_to_end_time;




        //Create the matrix


        //factorize the matrix

        //solve the matrix


  return 0;
}
