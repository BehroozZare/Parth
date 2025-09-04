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

bool solvingLaplacian(std::vector<int> &new_to_old_dof_mapping,
                      std::string csv_address, bool write_csv,
                      PARTH_SOLVER::LinSysSolver *solver) {
    double end_to_end_time = omp_get_wtime();
    // Find boundary edges
//    Eigen::MatrixXi E;
//    igl::boundary_facets(F, E);
//    // Find boundary vertices
    Eigen::VectorXi b, IA, IC;
//    igl::unique(E, b, IA, IC);
    Eigen::VectorXd z_coordinate = V.col(1);
    double min_Z = z_coordinate.minCoeff();
    double threshold = igl::avg_edge_length(V, F);
    std::vector<bool> is_boundary(V.rows(), false);
    std::vector<int> old_to_new_map(V.rows(), -1);
    int num_boundary = 0;
    for(int i = 0; i < z_coordinate.rows(); i++){
        if(z_coordinate(i) - min_Z < threshold){
            is_boundary[i] = true;
            num_boundary++;
        }
    }
    //insert all the indices of is_boundary[i] == true in b
    b.resize(num_boundary);
    int cnt = 0;
    for(int i = 0; i < is_boundary.size(); i++){
        if(is_boundary[i]){
            b[cnt] = i;
            cnt++;
        }
    }
    // List of all vertex indices
    Eigen::VectorXi all, in;
    igl::colon<int>(0, V.rows() - 1, all);
    // List of interior indices
    igl::setdiff(all, b, in, IA);
    if(b.rows() == 0){
        std::cout << "LAPLACIAN: No boundary found" << std::endl;
    }

    // Construct and slice up Laplacian
    Eigen::SparseMatrix<double> L, L_in_in, L_in_b;
    igl::cotmatrix(V, F, L);
    // Define decay rate
//    double lambda = 0.1; // Adjust this value as needed
//
//    // Create identity matrix
//    Eigen::SparseMatrix<double> I(V.rows(), V.rows());
//    I.setIdentity();
//
//    // Scale identity matrix by lambda
//    Eigen::SparseMatrix<double> lambdaI = lambda * I;
//
//    // Modify Laplacian: L + lambda * I
//    L = L + lambdaI;

    igl::slice(L, in, in, L_in_in);
    igl::slice(L, in, b, L_in_b);

    // Dirichlet boundary conditions from z-coordinate
    Z = V.col(2);
    Z.setConstant(5); //V.col(2);
    Eigen::VectorXd bc = Z(b);

    // Solve PDE
    Eigen::SparseMatrix<double> mesh_csc =
            PARTH::computeMeshFromHessian(L_in_in, DIM);

    Minus_L_in_in = -L_in_in;

    // Compute the mapping with boundary
    std::vector<int> parth_new_to_old_map(in.rows());
    Eigen::VectorXi in_inv(V.rows());
    in_inv.setOnes();
    in_inv = -in_inv;
    // computing in_prev_inv
    for (int i = 0; i < in.rows(); i++) {
        in_inv(in(i)) = i;
    }

    if (!new_to_old_dof_mapping.empty()) {
        for (int parth_dof_idx = 0; parth_dof_idx < in.rows(); parth_dof_idx++) {
            int curr_dof = in(parth_dof_idx);
            int prev_dof = new_to_old_dof_mapping[curr_dof];

            // If it is a newly added
            if (prev_dof == -1) {
                parth_new_to_old_map[parth_dof_idx] = -1;
                continue;
            }

            // If it is the first time that the function is called
            if (in_init_prev_inv.rows() == 0) {
                parth_new_to_old_map[parth_dof_idx] = parth_dof_idx;
                continue;
            }

            int curr_dof_pos = in_inv(curr_dof);
            int prev_dof_pos = in_init_prev_inv(prev_dof);
            assert(parth_dof_idx == curr_dof_pos);
            // If it existed but was not selected before
            if (prev_dof_pos == -1) {
                parth_new_to_old_map[parth_dof_idx] = -1;
                continue;
            }

            // If it existed and selected before
            parth_new_to_old_map[parth_dof_idx] = prev_dof_pos;
        }
    } else {
        parth_new_to_old_map.clear();
    }

    Eigen::VectorXd rhs;
    Eigen::VectorXd sol;
    Eigen::SparseMatrix<double> lower_A_csc = Minus_L_in_in.triangularView<Eigen::Lower>();
    solver->setFullCoeffMatrix(Minus_L_in_in);
    solver->setMatrix(lower_A_csc.outerIndexPtr(),
                      lower_A_csc.innerIndexPtr(), lower_A_csc.valuePtr(),
                      lower_A_csc.rows(), lower_A_csc.nonZeros(),
                      mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(),
                      mesh_csc.rows(), parth_new_to_old_map);

    // Analyze
    solver->analyze_pattern();

    // Factorize
    solver->factorize();

    // Solve
    rhs = L_in_b * bc;
    solver->solve(rhs, sol);
    Z(in) = sol;
    end_to_end_time = omp_get_wtime() - end_to_end_time;

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
    Z = Z.array().abs();
    double minZ = Z.minCoeff();
    double maxZ = Z.maxCoeff();
    Z = (Z.array() - minZ) / (maxZ - minZ);
    std::vector<int> global_group(V.rows(), 1);
    if(solver->parth.getReuse() != 0){
        for(auto& node: dirty_nodes){
            global_group[in[node]] = -1;
            //BLACK color for boundary nodes of Parth and grey color for inside nodes
            std::vector<int>& node_to_region = solver->parth.hmd.DOF_to_HMD_node;
            int* Mp = solver->parth.Mp;
            int* Mi = solver->parth.Mi;
            for(int nbr_ptr = Mp[node]; nbr_ptr < Mp[node + 1]; nbr_ptr++){
                int nbr = Mi[nbr_ptr];
//                Z(in(node)) = 1;
                if(node_to_region[nbr] != node_to_region[node]){
                    if(!dirty_flag[node_to_region[nbr]]){
                        Z(in(node)) = 1;
                        break;
                    }

                }
            }
        }
    }

    PARTH::applyTextureBasedOnLeavesAndSeparator(V, F, lvl,
                                                 global_group,
                                                 "/Users/behroozzare/Desktop/Graphics/ParthSolverDev/output/teaserOBJs/", step, 0);
    //Save the Z as a Eigen Vector in a text file
    std::string Z_address = "/Users/behroozzare/Desktop/Graphics/ParthSolverDev/output/teaserOBJs/Color" + std::to_string(step) + ".txt";
    std::ofstream Z_file(Z_address);
    if (Z_file.is_open()) {
        for (int i = 0; i < Z.rows(); i++) {
            Z_file << Z(i) << std::endl;
        }
        Z_file.close();
    } else {
        std::cerr << "Unable to open file";
    }

    // For adding to the record
    if (write_csv) {
        // Solve Quality
//        solver->residual =  (rhs - Minus_L_in_in_rowMajor.selfadjointView<Eigen::Upper>() * sol)
//                .norm();
        std::cout << "The residual is: " << solver->residual << std::endl;
        solver->printTiming();
        solver->addCSVRecord(csv_address, step++, 0, end_to_end_time);
    }

    // Reset the variables
    if (!init) {
        in_init_prev = in;
        in_init_prev_inv = in_inv;
        init = true;
    }

    return true;
}


void UpdateViewer(igl::opengl::glfw::Viewer &viewer) {
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    Eigen::MatrixXd C = Eigen::MatrixXd::Constant(F.rows(), 3, 1);
    // Compute the adjacency of a face
//    for (int i = 0; i < patch.rows(); i++) {
//        C.row(patch(i)) << 1, 0, 0;
//    }
//    viewer.data().set_colors(C);

    viewer.data().set_data(Z);

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

// GIF Wrapper
void initGif(igl::opengl::glfw::Viewer &viewer, std::string save_address) {
    GifBegin(&GIFWriter, save_address.c_str(),
             (viewer.core().viewport[2] - viewer.core().viewport[0]),
             (viewer.core().viewport[3] - viewer.core().viewport[1]), GIFDelay);
}

void writeGif(igl::opengl::glfw::Viewer &viewer) {
    int width =
            static_cast<int>((viewer.core().viewport[2] - viewer.core().viewport[0]));
    int height =
            static_cast<int>((viewer.core().viewport[3] - viewer.core().viewport[1]));

    // Allocate temporary buffers for image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

    // Draw the scene in the buffers
    viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);

    std::vector<uint8_t> img(width * height * 4);
    for (int rowI = 0; rowI < width; rowI++) {
        for (int colI = 0; colI < height; colI++) {
            int indStart = (rowI + (height - 1 - colI) * width) * 4;
            img[indStart] = R(rowI, colI);
            img[indStart + 1] = G(rowI, colI);
            img[indStart + 2] = B(rowI, colI);
            img[indStart + 3] = A(rowI, colI);
        }
    }

    GifWriteFrame(&GIFWriter, img.data(), width, height, GIFDelay);
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

    if (!igl::read_triangle_mesh(args.input, OV, OF)) {
        std::cerr << "Failed to read the mesh" << std::endl;
        return 0;
    }

    V = OV;
    F = OF;

//    //If there is no boundary
//    Eigen::MatrixXi E;
//    igl::boundary_facets(F, E);
//    if(E.rows() == 0){
//        //Detect the vertices with the smallest Z values (with a treshold)
//        Eigen::VectorXd z_coordinate = V.col(1);
//        double minZ = z_coordinate.minCoeff();
//        double threshold = igl::avg_edge_length(V, F);
//        std::vector<bool> delete_V_flag(V.rows(), false);
//        std::vector<int> old_to_new_map(V.rows(), -1);
//        int delete_v_cnt = 0;
//        for(int i = 0; i < z_coordinate.rows(); i++){
//            if(z_coordinate(i) - minZ < threshold){
//                delete_V_flag[i] = true;
//                delete_v_cnt++;
//            } else {
//                old_to_new_map[i] = i - delete_v_cnt;
//            }
//        }
//
//
//        //Delete the faces that have the deleted vertices
//        std::vector<bool> delete_F_flag(F.rows(), false);
//        int delete_f_cnt = 0;
//        for(int i = 0; i < F.rows(); i++){
//            for(int j = 0; j < F.cols(); j++){
//                if(delete_V_flag[F(i,j)]){
//                    delete_F_flag[i] = true;
//                    delete_f_cnt++;
//                    break;
//                }
//            }
//        }
//        //Create the new V and F
//        Eigen::MatrixXd V_new(V.rows() - delete_v_cnt, V.cols());
//        Eigen::MatrixXi F_new(F.rows() - delete_f_cnt, F.cols());
//        int cnt = 0;
//        for(int i = 0; i < V.rows(); i++){
//            if(!delete_V_flag[i]){
//                V_new.row(cnt) = V.row(i);
//                cnt++;
//            }
//        }
//
//        cnt = 0;
//        for(int i = 0; i < F.rows(); i++){
//            if(!delete_F_flag[i]){
//                //Update the vertex indices of the face
//                for(int j = 0; j < F.cols(); j++){
//                    F_new(cnt, j) = old_to_new_map[F(i,j)];
//                }
//                cnt++;
//            }
//        }
//        //Make F_new and V_new equal to F and V
//        V = V_new;
//        F = F_new;
//        OV = V;
//        OF = F;
//        //Check whether the boundary is created
//        igl::boundary_facets(F, E);
//        if(E.rows() == 0){
//            std::cerr << "The mesh does not have a boundary" << std::endl;
//            return 0;
//        }
//    }
//
//

    // Compute S random samples
    std::vector<int> fid_list{6695};
//    std::vector<int> fid_list{1995};
    int S = 50;
    // Get 50 uniform distributed fixed(not random) sample from [0, F.rows()]
    // range)

    // Different patch sizes
    std::vector<double> percentage{0.02};

    bool activate_parth = args.activate_parth;
    // Creating the solver
    std::string data_name = args.simName;
    PARTH_SOLVER::LinSysSolver *solver;

    solver = PARTH_SOLVER::LinSysSolver::create(
            PARTH_SOLVER::LinSysSolverType::ACCELERATE);


    if (solver == nullptr) {
        std::cerr << "Solver is not created" << std::endl;
        return 0;
    }

    // Solver Config
    solver->setSimulationDIM(1);
    solver->parth.activate_aggressive_reuse = false;
    solver->parth.lagging = false;
    solver->reorderingType = PARTH_SOLVER::ReorderingType::METIS;
//    solver->parth.setNDLevels(4);

    std::vector<int> new_to_old_map;
    new_to_old_map.clear();
    solver->setParthActivation(activate_parth);
    solvingLaplacian(new_to_old_map, args.simName, true, solver);
    // Main loop for experiment -> Parth Timing
    for (int r_size = 0; r_size < percentage.size(); r_size++) {
        for (int test_cnt = 0; test_cnt < fid_list.size(); test_cnt++) {
            std::cout << "----------------- Test number: " << test_cnt
                      << " -----------------" << std::endl;
            //------------------- Randomly select a Patch -------------------
            PARTHDEMO::createPatch(fid_list[test_cnt], percentage[r_size], patch, F,
                                   V);
            Eigen::MatrixXi F_up;
            Eigen::MatrixXd V_up;

            std::vector<int> new_to_old_map_per_iter;
            for(int i = 0; i < 4; i++){
                patch = PARTHDEMO::remesh(patch, F, V, F_up, V_up, args.scale * 3,
                                          new_to_old_map_per_iter);
                if(new_to_old_map.empty()){
                    new_to_old_map = new_to_old_map_per_iter;
                } else {
                    for(int j = 0; j < new_to_old_map_per_iter.size();j++){
                        if(new_to_old_map_per_iter[j] != -1){
                            new_to_old_map_per_iter[j] = new_to_old_map[new_to_old_map_per_iter[j]];
                        }
                    }
                    new_to_old_map = new_to_old_map_per_iter;
                }
                new_to_old_map_per_iter.clear();
                V = V_up;
                F = F_up;
            }

            // Solving the laplacian
            solvingLaplacian(new_to_old_map, args.simName, true, solver);
            new_to_old_map.clear();
        }
    }

    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_pressed =
            [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int mods) -> bool {
                switch (key) {
                    default:
                        return false;
                    case 'r':
                        std::cout << "Reseting the mesh" << std::endl;
                        reset();
                        UpdateViewer(viewer);
                        return true;
                }
                return false;
            };

//
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        int fid;
        Eigen::Vector3f bc;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        igl::embree::EmbreeIntersector ei;
        ei.init(V.cast<float>(),F);
        Eigen::VectorXi Fp;
        Eigen::VectorXi Fi;
        igl::vertex_triangle_adjacency(F, V.rows(), Fi, Fp);
        // Initialize white
        Eigen::MatrixXd C = Eigen::MatrixXd::Constant(F.rows(),3,1);
        if(igl::embree::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core().view,
                                            viewer.core().proj, viewer.core().viewport, ei, fid, bc))
        {
            patch.resize(0);
            std::cout << "FID: " << fid << std::endl;
            PARTHDEMO::createPatch(fid, 0.05, patch, F, V);
//            Eigen::MatrixXd V_up;
//            Eigen::MatrixXi F_up;
//            patch = PARTHDEMO::remesh(patch, F, V, F_up, V_up, args.scale * 2,
//                                      new_to_old_map);
//            V = V_up;
//            F = F_up;
//            patch = PARTHDEMO::remesh(patch, F, V, F_up, V_up, args.scale * 2,
//                                      new_to_old_map);
//            V = V_up;
//            F = F_up;
//            patch = PARTHDEMO::remesh(patch, F, V, F_up, V_up, args.scale * 2,
//                                      new_to_old_map);

            UpdateViewer(viewer);
            return true;
        }
        return false;
    };



    viewer.core().is_animating = true;
    UpdateViewer(viewer);
    viewer.core().is_animating = false;
    viewer.launch();


    delete solver;
    return 0;
}