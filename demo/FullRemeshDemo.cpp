//
// Created by behrooz zare on 2024-04-28.
//
//
// Created by behrooz zare on 2024-04-06.
//
//igl include
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/unique.h>
#include <igl/setdiff.h>
#include <igl/cotmatrix.h>
#include <igl/avg_edge_length.h>


#include "LinSysSolver.hpp"
#include "Parth_utils.h"

#include "upSample.h"
#include "createPatch.h"
#include "remesh_botsch.h"

#include "GIF.h"
#include "CLI/CLI.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <iostream>



#include <algorithm>
#include <omp.h>
#include "projected_cotmatrix.h"

Eigen::MatrixXd OV, V;
Eigen::MatrixXi OF, F;
Eigen::VectorXd Z;
bool init = false;
Eigen::VectorXi in_init_prev_inv;
Eigen::VectorXi in_init_prev;
Eigen::VectorXi patch;
int step = 0;
int DIM = 1;

// Gif variables
GifWriter GIFWriter;
uint32_t GIFDelay = 100; //*10ms


void reset() {
    V = OV;
    F = OF;
    patch.resize(0);
}


void remesh(Eigen::MatrixXi &F, Eigen::MatrixXd &V, double scale, std::vector<int>& new_to_old_dof_map){
    double h = igl::avg_edge_length(V, F);
    Eigen::VectorXd target;
    target = Eigen::VectorXd::Constant(V.rows(),h * scale);
    Eigen::VectorXi feature;
    feature.resize(0);
    std::vector<int> old_to_new_dof_map;
    remesh_botsch_map(V, F, old_to_new_dof_map, target, 10, feature, false);
    //computing new_to_old_dof_map;
    new_to_old_dof_map.clear();
    new_to_old_dof_map.resize(V.rows(), -1);
    for(int i = 0; i < old_to_new_dof_map.size(); i++){
        assert(old_to_new_dof_map[i] < V.rows());
        if(old_to_new_dof_map[i] != -1){
            new_to_old_dof_map[old_to_new_dof_map[i]] = i;
        }
    }
}

bool solvingLaplacian(std::vector<int> &new_to_old_dof_mapping,
                      std::string csv_address,
                      bool write_csv,
                      PARTH_SOLVER::LinSysSolver* solver) {
    double end_to_end_time = omp_get_wtime();
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
    PARTHDEMO::projected_cotmatrix(V, F, L);
    igl::slice(L, in, in, L_in_in);
    igl::slice(L, in, b, L_in_b);

    // Dirichlet boundary conditions from z-coordinate
    Z = V.col(2);
    Eigen::VectorXd bc = Z(b);

    // Solve PDE
    Eigen::SparseMatrix<double> mesh_csc =
            PARTH::computeMeshFromHessian(L_in_in, DIM);

    Eigen::SparseMatrix<double> Minus_L_in_in = -L_in_in;


    //Compute the mapping with boundary
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

    //For adding to the record
    if(write_csv){
        // Solve Quality
        solver->residual = (rhs - Minus_L_in_in.selfadjointView<Eigen::Lower>() * sol).norm();
        solver->printTiming();
        solver->addCSVRecord(csv_address, 0, step++, end_to_end_time);
    }

    //Reset the variables
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
    //Compute the adjacency of a face
    for (int i = 0; i < patch.rows(); i++) {
        C.row(patch(i)) << 1, 0, 0;
    }
    viewer.data().set_colors(C);


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
    GifBegin(&GIFWriter,save_address.c_str(),
             (viewer.core().viewport[2] - viewer.core().viewport[0]),
             (viewer.core().viewport[3] - viewer.core().viewport[1]), GIFDelay);
}

void writeGif(igl::opengl::glfw::Viewer &viewer) {
    int width = static_cast<int>((viewer.core().viewport[2] - viewer.core().viewport[0]));
    int height = static_cast<int>((viewer.core().viewport[3] - viewer.core().viewport[1]));

    // Allocate temporary buffers for image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

    // Draw the scene in the buffers
    viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);

    std::vector <uint8_t> img(width * height * 4);
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



int main(int argc, char *argv[]) {

    CLIArgs args(argc, argv);

    if(!igl::read_triangle_mesh(args.input, OV, OF)){
        std::cerr << "Failed to read the mesh" << std::endl;
        return 0;
    }

    V = OV;
    F = OF;

    //Creating the solver
    std::string parth_data_name;
    std::string basic_data_name;
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

    if(solver== nullptr){
        std::cerr << "Solver is not created" << std::endl;
        return 0;
    }
    //Solver Config
    solver->setSimulationDIM(1);

    std::vector<int> tmp;
    solver->setParthActivation(true);
    solvingLaplacian(tmp, parth_data_name, true, solver);

    int TOTAL_TEST = 5;

    //Main loop for experiment -> Parth Timing
    for(int test_cnt = 0; test_cnt < TOTAL_TEST; test_cnt++){
        std::cout << "----------------- Test number: " << test_cnt << " -----------------" << std::endl;
        solver->resetParth();
        solvingLaplacian(tmp, basic_data_name, false, solver); //Init Parth datastructure

        //------------------- Fix the quality of the mesh -------------------
        std::vector<int> new_to_old_map;
        remesh(F, V, args.scale, new_to_old_map);
        //Solving the laplacian
        solvingLaplacian(new_to_old_map, basic_data_name, true, solver);
    }

    solver->resetTotalTime();
    solver->resetParth();
    reset();

    solver->setParthActivation(false);
    tmp.clear();
    solvingLaplacian(tmp, basic_data_name, true, solver);
    //Main loop for experiment -> Parth Timing
    for(int test_cnt = 0; test_cnt < TOTAL_TEST; test_cnt++){
        std::cout << "----------------- Test number: " << test_cnt << " -----------------" << std::endl;
        reset();
        solvingLaplacian(tmp, basic_data_name, false, solver); //Init Parth datastructure

        //------------------- Fix the quality of the mesh -------------------
        std::vector<int> new_to_old_map;
        remesh(F, V, args.scale, new_to_old_map);
        //Solving the laplacian
        solvingLaplacian(new_to_old_map, basic_data_name, true, solver);
    }
    delete solver;




    igl::opengl::glfw::Viewer viewer;

    int frame = 0;
    reset();
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
        if (frame == TOTAL_TEST) {
            viewer.core().is_animating = false;
        } else {
            viewer.core().is_animating = true;
        }
        if (viewer.core().is_animating) {
            UpdateViewer(viewer);
        }
        frame++;

        return false;
    };

    bool init_gif = false;
    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
        if (viewer.core().is_animating) {
            std::vector<int> new_to_old_map;
            remesh(F, V, args.scale, new_to_old_map);
            if (!init_gif) {
                initGif(viewer, args.output + "out.gif");
                init_gif = true;
            }
            writeGif(viewer);
        } else {
            GifEnd(&GIFWriter);
            return true;
        }
        return false;
    };


    UpdateViewer(viewer);
    viewer.core().is_animating = true;
    viewer.launch();
    return 0;
}