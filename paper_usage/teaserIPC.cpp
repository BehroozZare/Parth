//
// Created by behrooz zare on 2024-04-28.
//
// Use the meshes in https://github.com/odedstein/meshes

// igl include
#include <igl/cotmatrix.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/setdiff.h>
#include <igl/unique.h>
#include <igl/embree/unproject_onto_mesh.h>
#include <igl/boundary_loop.h>
#include <igl/avg_edge_length.h>
#include <unsupported/Eigen/SparseExtra>


#include "LinSysSolver.hpp"
#include "Parth_utils.h"

#include "createPatch.h"
#include "remeshPatch.h"
#include "upSample.h"

#include "CLI/CLI.hpp"
#include "GIF.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <iostream>

#include "projected_cotmatrix.h"
#include <algorithm>
#include <omp.h>

Eigen::MatrixXd OV, V;
Eigen::MatrixXi OF, F;
Eigen::SparseMatrix<double> S;
Eigen::VectorXd Z;


bool init = false;
Eigen::VectorXi in_init_prev_inv;
Eigen::VectorXi in_init_prev;
Eigen::VectorXi patch;
int step = 0;
int DIM = 1;

Eigen::SparseMatrix<double> Minus_L_in_in;

// Gif variables
GifWriter GIFWriter;
uint32_t GIFDelay = 100; //*10ms

void reset() {
    V = OV;
    F = OF;
    patch.resize(0);
}

bool teaserReuse(PARTH_SOLVER::LinSysSolver *solver) {
    double end_to_end_time = omp_get_wtime();

    std::vector<int> new_to_old_map;
    new_to_old_map.clear();
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);
    std::cout << "Edges: " << L.nonZeros() << std::endl;
    Eigen::SparseMatrix<double> lower_A_csc = L.triangularView<Eigen::Lower>();
    Eigen::SparseMatrix<double> mesh_csc = PARTH::computeMeshFromHessian(L, DIM);
    solver->setFullCoeffMatrix(L);
    solver->setMatrix(lower_A_csc.outerIndexPtr(),
                      lower_A_csc.innerIndexPtr(), lower_A_csc.valuePtr(),
                      lower_A_csc.rows(), lower_A_csc.nonZeros(),
                      mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(),
                      mesh_csc.rows(), new_to_old_map);

    // Analyze
    solver->analyze_pattern();

    // Factorize
    solver->factorize();

    //Add the appropriate texture to the obj file
    int total_num_groups = solver->parth.hmd.num_HMD_nodes;
    int lvl = solver->parth.hmd.num_levels;
    std::vector<int> dirty_nodes = solver->parth.integrator.getDirtyNodes(solver->parth.hmd);
    std::vector<int> fine_dirty = solver->parth.integrator.dirty_fine_HMD_nodes_saved;
    std::vector<int> coarse_dirty = solver->parth.integrator.dirty_coarse_HMD_nodes_saved;
    solver->parth.integrator.dirty_fine_HMD_nodes_saved.clear();
    solver->parth.integrator.dirty_coarse_HMD_nodes_saved.clear();
    std::vector<int> group(solver->parth.M_n, 1);
    std::vector<bool> dirty_flag(solver->parth.hmd.num_HMD_nodes, false);
    for(auto& node: fine_dirty){
        dirty_flag[node] = true;
    }
    //normalize absolute value of Z based on its absolute min and max
    Z.resize(V.rows());
    std::vector<int> global_group(V.rows(), 1);
    if(solver->parth.getReuse() != 0){
        for(auto& node: dirty_nodes){
            global_group[node] = -1;
            //BLACK color for boundary nodes of Parth and grey color for inside nodes
            std::vector<int>& node_to_region = solver->parth.hmd.DOF_to_HMD_node;
            int* Mp = solver->parth.Mp;
            int* Mi = solver->parth.Mi;
            for(int nbr_ptr = Mp[node]; nbr_ptr < Mp[node + 1]; nbr_ptr++){
                int nbr = Mi[nbr_ptr];
//                Z(in(node)) = 1;
                if(node_to_region[nbr] != node_to_region[node]){
                    if(!dirty_flag[node_to_region[nbr]]){
                        Z(node) = 1;
                        break;
                    }

                }
            }
        }
    }

//    PARTH::applyTextureBasedOnLeavesAndSeparator(V, F, lvl,
//                                                 global_group,
//                                                 "/Users/behroozzare/Desktop/Graphics/ParthSolverDev/output/teaserOBJs/", step, 0);
//    //Save the Z as a Eigen Vector in a text file
//    std::string Z_address = "/Users/behroozzare/Desktop/Graphics/ParthSolverDev/output/teaserOBJs/Color" + std::to_string(step) + ".txt";
//    std::ofstream Z_file(Z_address);
//    if (Z_file.is_open()) {
//        for (int i = 0; i < Z.rows(); i++) {
//            Z_file << Z(i) << std::endl;
//        }
//        Z_file.close();
//    } else {
//        std::cerr << "Unable to open file";
//    }


    std::cout << "The residual is: " << solver->residual << std::endl;
    solver->printTiming();

    return true;
}


void UpdateViewer(igl::opengl::glfw::Viewer &viewer) {
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    Eigen::MatrixXd C = Eigen::MatrixXd::Constant(F.rows(), 3, 1);
    // Compute the adjacency of a face
    for (int i = 0; i < patch.rows(); i++) {
        C.row(patch(i)) << 1, 0, 0;
    }
    viewer.data().set_colors(C);

//    viewer.data().set_data(Z);

    double zoom = 1.0;
    viewer.core().background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.data().show_lines = true;
    viewer.core().lighting_factor = 1.0;
    viewer.data().show_texture = true;
    viewer.core().orthographic = false;
    viewer.core().camera_zoom *= zoom;
    viewer.core().animation_max_fps = 2.0;
    viewer.data().point_size = 5;
    viewer.data().show_overlay = true;

    //    viewer.core().trackball_angle = Eigen::Quaternionf(
    //            Eigen::AngleAxisf(M_PI_4 / 2.0, Eigen::Vector3f::UnitX()));
}

namespace fs = std::filesystem;

struct CLIArgs {
    std::string input;
    std::string output;
    std::string simName = "test";
    std::string order = "METIS";
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
        app.add_option("--Order", order, "ADD PARTH IMMOBILIZER");
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

int main(int argc, char *argv[]) {

    CLIArgs args(argc, argv);

    bool activate_parth = args.activate_parth;
    std::string data_name = args.simName;

    // Creating the solver
    PARTH_SOLVER::LinSysSolver *solver;

    solver = PARTH_SOLVER::LinSysSolver::create(
            PARTH_SOLVER::LinSysSolverType::ACCELERATE);


    if (solver == nullptr) {
        std::cerr << "Solver is not created" << std::endl;
        return 0;
    }

    for(auto& i : {6, 20}){
        //Read The mesh
        if (!igl::read_triangle_mesh(args.input + "in_" + std::to_string(i) + "_0.obj", V, F)) {
            std::cerr << "Failed to read the mesh" << std::endl;
            return 0;
        } else {
            std::cout << "Mesh in address: " << args.input + "in_" + std::to_string(i) + "_0.obj" << " is read successfully" << std::endl;
        }
        //Alternatively build the mesh from laplacian
        // Eigen::SparseMatrix<double> L;
        // igl::cotmatrix(V, F, L);
        // Eigen::SparseMatrix<double> lower_A_csc = L.triangularView<Eigen::Lower>();

        //Read the Hessian
        std::string hessian_name =
                args.input + "hessian_" + std::to_string(i) + "_" +
                std::to_string(0) + "_" + "last_IPC.mtx";

        Eigen::SparseMatrix<double> lower_A_csc;
        if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
            std::cerr << "File " << hessian_name << " is not found" << std::endl;
            continue;
        }
        assert(lower_A_csc.rows() == 3 * V.rows());

        // Solver Config
        int dim = 3;
        solver->setSimulationDIM(dim);
        solver->parth.activate_aggressive_reuse = false;
        solver->parth.lagging = false;
        solver->reorderingType = PARTH_SOLVER::ReorderingType::METIS;
        solver->setParthActivation(activate_parth);
        std::vector<int> new_to_old_map;
        Eigen::SparseMatrix<double> mesh_csc = PARTH::computeMeshFromHessian(lower_A_csc, dim);
        solver->setFullCoeffMatrix(lower_A_csc);
        solver->setMatrix(lower_A_csc.outerIndexPtr(),
                          lower_A_csc.innerIndexPtr(), lower_A_csc.valuePtr(),
                          lower_A_csc.rows(), lower_A_csc.nonZeros(),
                          mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(),
                          mesh_csc.rows(), new_to_old_map);

        // Analyze
        solver->analyze_pattern();

        // Factorize
        solver->factorize();

        int total_num_groups = solver->parth.hmd.num_HMD_nodes;
        int lvl = solver->parth.hmd.num_levels;
        std::vector<int> dirty_nodes = solver->parth.integrator.getDirtyNodes(solver->parth.hmd);
        std::vector<int> fine_dirty = solver->parth.integrator.dirty_fine_HMD_nodes_saved;
        std::vector<int> coarse_dirty = solver->parth.integrator.dirty_coarse_HMD_nodes_saved;
        solver->parth.integrator.dirty_fine_HMD_nodes_saved.clear();
        solver->parth.integrator.dirty_coarse_HMD_nodes_saved.clear();
        std::vector<int> group(solver->parth.M_n, 1);
        std::vector<bool> dirty_flag(solver->parth.hmd.num_HMD_nodes, false);
        for(auto& node: fine_dirty){
            dirty_flag[node] = true;
        }

        Z.resize(V.rows());
        Z.setZero();
        if(solver->parth.getReuse() != 0){
            for(auto& node: dirty_nodes){
                group[node] = -1;
                //BLACK color for boundary nodes of Parth and grey color for inside nodes
                std::vector<int>& node_to_region = solver->parth.hmd.DOF_to_HMD_node;
                int* Mp = solver->parth.Mp;
                int* Mi = solver->parth.Mi;
                for(int nbr_ptr = Mp[node]; nbr_ptr < Mp[node + 1]; nbr_ptr++){
                    int nbr = Mi[nbr_ptr];
//                Z(in(node)) = 1;
                    if(node_to_region[nbr] != node_to_region[node]){
                        if(!dirty_flag[node_to_region[nbr]]){
                            Z(node) = 1;
                            break;
                        }

                    }
                }
            }
        }

        PARTH::applyTextureBasedOnLeavesAndSeparator(V, F, lvl,
                                             group,
                                             args.input, i, 0);
        
        std::cout << "The residual is: " << solver->residual << std::endl;
        solver->printTiming();

    }

    delete solver;
    return 0;
}