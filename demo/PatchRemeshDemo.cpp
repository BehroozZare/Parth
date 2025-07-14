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
//Eigen::VectorXi in_init_prev_inv;
//Eigen::VectorXi in_init_prev;
Eigen::VectorXi patch;
int step = 0;
int DIM = 1;

//Eigen::SparseMatrix<double> Minus_L_in_in;
//Eigen::SparseMatrix<double, Eigen::RowMajor> Minus_L_in_in_rowMajor;
//
// Gif variables
GifWriter GIFWriter;
uint32_t GIFDelay = 100; //*10ms

void reset() {
  V = OV;
  F = OF;
  patch.resize(0);
}

//bool solvingLaplacian(std::vector<int> &new_to_old_dof_mapping,
//                      std::string csv_address, bool write_csv,
//                      PARTH_SOLVER::LinSysSolver *solver) {
//  double end_to_end_time = omp_get_wtime();
//  // Find boundary edges
//  Eigen::MatrixXi E;
//  igl::boundary_facets(F, E);
//  // Find boundary vertices
//  Eigen::VectorXi b, IA, IC;
//  igl::unique(E, b, IA, IC);
//  // List of all vertex indices
//  Eigen::VectorXi all, in;
//  igl::colon<int>(0, V.rows() - 1, all);
//  // List of interior indices
//  igl::setdiff(all, b, in, IA);
//
//  // Construct and slice up Laplacian
//  Eigen::SparseMatrix<double> L, L_in_in, L_in_b;
//  PARTHDEMO::projected_cotmatrix(V, F, L);
//  igl::slice(L, in, in, L_in_in);
//  igl::slice(L, in, b, L_in_b);
//
//  // Dirichlet boundary conditions from z-coordinate
//  Z = V.col(2);
//  Eigen::VectorXd bc = Z(b);
//
//  // Solve PDE
//  Eigen::SparseMatrix<double> mesh_csc =
//      PARTH::computeMeshFromHessian(L_in_in, DIM);
//
//  Minus_L_in_in = -L_in_in;
//
//  // Compute the mapping with boundary
//  std::vector<int> parth_new_to_old_map(in.rows());
//  Eigen::VectorXi in_inv(V.rows());
//  in_inv.setOnes();
//  in_inv = -in_inv;
//  // computing in_prev_inv
//  for (int i = 0; i < in.rows(); i++) {
//    in_inv(in(i)) = i;
//  }
//
//  if (!new_to_old_dof_mapping.empty()) {
//    for (int parth_dof_idx = 0; parth_dof_idx < in.rows(); parth_dof_idx++) {
//      int curr_dof = in(parth_dof_idx);
//      int prev_dof = new_to_old_dof_mapping[curr_dof];
//
//      // If it is a newly added
//      if (prev_dof == -1) {
//        parth_new_to_old_map[parth_dof_idx] = -1;
//        continue;
//      }
//
//      // If it is the first time that the function is called
//      if (in_init_prev_inv.rows() == 0) {
//        parth_new_to_old_map[parth_dof_idx] = parth_dof_idx;
//        continue;
//      }
//
//      int curr_dof_pos = in_inv(curr_dof);
//      int prev_dof_pos = in_init_prev_inv(prev_dof);
//      assert(parth_dof_idx == curr_dof_pos);
//      // If it existed but was not selected before
//      if (prev_dof_pos == -1) {
//        parth_new_to_old_map[parth_dof_idx] = -1;
//        continue;
//      }
//
//      // If it existed and selected before
//      parth_new_to_old_map[parth_dof_idx] = prev_dof_pos;
//    }
//  } else {
//    parth_new_to_old_map.clear();
//  }
//
//  Eigen::VectorXd rhs;
//  Eigen::VectorXd sol;
//  Eigen::SparseMatrix<double> lower_A_csc = Minus_L_in_in.triangularView<Eigen::Lower>();
//  solver->setFullCoeffMatrix(Minus_L_in_in);
//  solver->setMatrix(lower_A_csc.outerIndexPtr(),
//                    lower_A_csc.innerIndexPtr(), lower_A_csc.valuePtr(),
//                    lower_A_csc.rows(), lower_A_csc.nonZeros(),
//                    mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(),
//                    mesh_csc.rows(), parth_new_to_old_map);
//
//  // Analyze
//  solver->analyze_pattern();
//
//  // Factorize
//  solver->factorize();
//
//  // Solve
//  rhs = L_in_b * bc;
//  solver->solve(rhs, sol);
//  Z(in) = sol;
//  end_to_end_time = omp_get_wtime() - end_to_end_time;
//
//  // For adding to the record
//  if (write_csv) {
//    // Solve Quality
//    if (solver->type() == PARTH_SOLVER::LinSysSolverType::MKL) {
//      solver->residual =
//          (rhs - Minus_L_in_in_rowMajor.selfadjointView<Eigen::Upper>() * sol)
//              .norm();
//    } else {
//      solver->residual =
//          (rhs - Minus_L_in_in.selfadjointView<Eigen::Lower>() * sol).norm();
//    }
//    solver->printTiming();
//    solver->addCSVRecord(csv_address, 0, step++, end_to_end_time);
//  }
//
//  // Reset the variables
//  if (!init) {
//    in_init_prev = in;
//    in_init_prev_inv = in_inv;
//    init = true;
//  }
//
//  return true;
//}

bool Smoothing(std::vector<int> &new_to_old_dof_mapping,
                      std::string csv_address, bool write_csv,
                      PARTH_SOLVER::LinSysSolver *solver) {
  double end_to_end_time = omp_get_wtime();

  Eigen::SparseMatrix<double> L;
  // Compute Laplace-Beltrami operator: #V by #V
  PARTHDEMO::projected_cotmatrix(V, F, L);

  // Recompute just mass matrix on each step
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  // Solve (M-delta*L) U = M*U
  S = (M - 0.001 * L);

  Eigen::SparseMatrix<double> mesh_csc = PARTH::computeMeshFromHessian(S, DIM);

  Eigen::SparseMatrix<double> lower_A_csc = S.triangularView<Eigen::Lower>();
  solver->setFullCoeffMatrix(S);
  solver->setMatrix(lower_A_csc.outerIndexPtr(),
                    lower_A_csc.innerIndexPtr(), lower_A_csc.valuePtr(),
                    lower_A_csc.rows(), lower_A_csc.nonZeros(),
                    mesh_csc.outerIndexPtr(), mesh_csc.innerIndexPtr(),
                    mesh_csc.rows(), new_to_old_dof_mapping);

  // Analyze
  solver->analyze_pattern();

  // Factorize
  solver->factorize();

  // Solve
  Eigen::MatrixXd rhs = M * V;
  Eigen::MatrixXd tmp = V;
  solver->solve(rhs, V);


  end_to_end_time = omp_get_wtime() - end_to_end_time;


  // For adding to the record
  if (write_csv) {
    // Solve Quality
    solver->residual =  (S * V - M * tmp).norm();
    std::cout << "The residual is: " << solver->residual << std::endl;
    solver->printTiming();
    solver->addCSVRecord(csv_address, step++, 0, end_to_end_time);
  }

  // Compute centroid and subtract (also important for numerics)
  Eigen::VectorXd dblA;
  igl::doublearea(V, F, dblA);
  double area = 0.5 * dblA.sum();

  Eigen::MatrixXd BC;
  igl::barycenter(V, F, BC);
  Eigen::RowVector3d centroid(0, 0, 0);
  for (int i = 0; i < BC.rows(); i++) {
    centroid += 0.5 * dblA(i) / area * BC.row(i);
  }
  V.rowwise() -= centroid;
  // Normalize to unit surface area (important for numerics)
  V.array() /= sqrt(area);

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

  // Compute S random samples
  std::vector<int> fid_list;
  int S = 50;
  // Get 50 uniform distributed fixed(not random) sample from [0, F.rows()]
  // range)
  for (int i = 0; i < S; i++) {
    fid_list.push_back(i * F.rows() / S);
  }
  // Different patch sizes
  std::vector<double> percentage{0.01, 0.05, 0.1, 0.2};

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
  } else if (args.solver_type == "PARSY") {
          solver =
              PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::SYMPILER);
  } else if (args.solver_type == "EIGEN"){
          solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::EIGEN);
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
  } else if (args.solver_type == "PARALLEL_LAZY_CHOLMOD") {
    solver = PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::PARALLEL_LAZY_CHOLMOD);
  }else {
    std::cerr << "UNKNOWN linear solver" << std::endl;
    return 0;
  }

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

  std::vector<int> new_to_old_map;
  new_to_old_map.clear();
  solver->setParthActivation(activate_parth);
  Smoothing(new_to_old_map, args.simName, true, solver);
  // Main loop for experiment -> Parth Timing
  for (int r_size = 0; r_size < percentage.size(); r_size++) {
    for (int test_cnt = 0; test_cnt < fid_list.size(); test_cnt++) {
      std::cout << "----------------- Test number: " << test_cnt
                << " -----------------" << std::endl;
      reset();
      solver->resetSolver();
      new_to_old_map.clear();
      Smoothing(new_to_old_map, args.simName, false,
                       solver); // Init Parth datastructure
      Eigen::MatrixXi F_up;
      Eigen::MatrixXd V_up;

      //------------------- Randomly select a Patch -------------------
      PARTHDEMO::createPatch(fid_list[test_cnt], percentage[r_size], patch, F,
                             V);
      std::cout << "The Patch ratio is: " << patch.size() * 1.0 / F.rows()
                << std::endl;
      assert(patch.rows() != 0);
      //------------------- Increase the dofs in that patch -------------------
      if (args.patch == "UPSAMPLE") {
        patch = PARTHDEMO::upSampleThePatch(patch, F, V, F_up, V_up,
                                            new_to_old_map);
      } else if (args.patch == "REMESH") {
        patch = PARTHDEMO::remesh(patch, F, V, F_up, V_up, args.scale,
                                  new_to_old_map);
      } else {
        std::cerr << "UNKNOWN patching strategy" << std::endl;
      }

      V = V_up;
      F = F_up;
      // Solving the laplacian
      Smoothing(new_to_old_map, args.simName, true, solver);
    }
  }

//  igl::opengl::glfw::Viewer viewer;
//
//  int frame = 0;
//  int ring_size = 0;
//  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) ->
//  bool {
//      if (frame == fid_list.size()) {
//          viewer.core().is_animating = false;
//      } else {
//          viewer.core().is_animating = true;
//      }
//      if (viewer.core().is_animating) {
//          Eigen::MatrixXi F_up;
//          Eigen::MatrixXd V_up;
//          //------------------- Randomly select a Patch -------------------
//          PARTHDEMO::createPatch(fid_list[frame],
//          percentage[ring_size++], patch, F, V);
//          //------------------- Increase the dofs in that patch -------------------
//         std::vector<int> new_to_old_dof_map;
//
//          patch = PARTHDEMO::remesh(patch, F, V, F_up, V_up, args.scale,
//          new_to_old_dof_map); V = V_up; F = F_up;
//
//          UpdateViewer(viewer);
//
//          if(ring_size % percentage.size() == 0){
//              frame++;
//              ring_size = 0;
//          }
//      }
//
//      return false;
//  };
//
//  bool init_gif = false;
//  viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &viewer) ->
//  bool {
//      if (viewer.core().is_animating) {
//          if (!init_gif) {
//              initGif(viewer, args.output + "out.gif");
//              init_gif = true;
//          }
//          writeGif(viewer);
//      } else {
//          GifEnd(&GIFWriter);
//          return true;
//      }
//      F = OF;
//      V = OV;
//      return false;
//  };
//
//
//  UpdateViewer(viewer);
//  viewer.core().is_animating = true;
//  viewer.launch();

  delete solver;
  return 0;
}