//
// Created by behrooz zare on 2024-05-06.
//

//
// Created by behrooz zare on 2024-04-28.
//
//
// Created by behrooz zare on 2024-04-06.
//
// igl include
#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/setdiff.h>
#include <igl/unique.h>

#include "LinSysSolver.hpp"
#include "Parth_utils.h"

#include "createPatch.h"
#include "remesh_botsch.h"
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
Eigen::SparseMatrix<double, Eigen::RowMajor> SRowMajor;
Eigen::SparseMatrix<double> L;
double h;
int DIM = 1;
int step = 0;
bool show_vis = false;

// Gif variables
GifWriter GIFWriter;
uint32_t GIFDelay = 100; //*10ms

void reset() {
  V = OV;
  F = OF;
  step = 0;
}

void remesh(Eigen::MatrixXi &F, Eigen::MatrixXd &V, double scale,
            std::vector<int> &new_to_old_dof_map) {
  h= igl::avg_edge_length(V, F);
  Eigen::VectorXd target;
  target = Eigen::VectorXd::Constant(V.rows(), h * scale);
  Eigen::VectorXi feature;
  feature.resize(0);
  std::vector<int> old_to_new_dof_map;
  std::cout << "_____ REMESH START _____" << std::endl;
  remesh_botsch_map(V, F, old_to_new_dof_map, target, 10, feature, false);
  std::cout << "_____ REMESH DONE _____" << std::endl;
  // computing new_to_old_dof_map;
  new_to_old_dof_map.clear();
  new_to_old_dof_map.resize(V.rows(), -1);
  for (int i = 0; i < old_to_new_dof_map.size(); i++) {
    assert(old_to_new_dof_map[i] < V.rows());
    if (old_to_new_dof_map[i] != -1) {
      new_to_old_dof_map[old_to_new_dof_map[i]] = i;
    }
  }
}

void Smoothing(std::vector<int> &new_to_old_dof_mapping,
               std::string csv_address, bool write_csv,
               PARTH_SOLVER::LinSysSolver *solver) {
  if (!solver) {
    std::cerr << "Solver is not created" << std::endl;
    return;
  }
  double end_to_end_time = omp_get_wtime();
  // Compute Laplace-Beltrami operator: #V by #V
  igl::cotmatrix(V, F, L);

  // Recompute just mass matrix on each step
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
  // Solve (M-delta*L) U = M*U
  S = (M - 0.001 * L);

  //    Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver1(S);
  //    assert(solver1.info() == Eigen::Success);
  //    V = solver1.solve(M*V).eval();

  Eigen::SparseMatrix<double> mesh_csc = PARTH::computeMeshFromHessian(S, DIM);

  if (solver->type() == PARTH_SOLVER::LinSysSolverType::MKL) {
    // Convert S from column major to row major
    SRowMajor = S;
    // Get the Upper triangle part of SRowMajor
    SRowMajor = SRowMajor.triangularView<Eigen::Upper>();
#ifndef NDEBUG
    int *Sp = SRowMajor.outerIndexPtr();
    int *Si = SRowMajor.innerIndexPtr();
    bool has_upper = false;
    bool has_lower = false;
    for (int i = SRowMajor.rows() / 2; i < SRowMajor.rows(); i++) {
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
      return;
    } else if (has_upper) {
      std::cout << "Upper" << std::endl;
    } else if (has_lower) {
      std::cerr << "Lower" << std::endl;
      return;
    }
#endif
    solver->setMatrix(SRowMajor.outerIndexPtr(), SRowMajor.innerIndexPtr(),
                      SRowMajor.valuePtr(), SRowMajor.rows(),
                      SRowMajor.nonZeros(), mesh_csc.outerIndexPtr(),
                      mesh_csc.innerIndexPtr(), mesh_csc.rows(),
                      new_to_old_dof_mapping);
  } else {
    solver->setMatrix(S.outerIndexPtr(), S.innerIndexPtr(), S.valuePtr(),
                      S.rows(), S.nonZeros(), mesh_csc.outerIndexPtr(),
                      mesh_csc.innerIndexPtr(), mesh_csc.rows(),
                      new_to_old_dof_mapping);
  }

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
    solver->printTiming();
    solver->addCSVRecord(csv_address, 0, step++, end_to_end_time);
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


}

void UpdateViewer(igl::opengl::glfw::Viewer &viewer) {
  viewer.data().clear();
  viewer.data().set_mesh(V, F);
  viewer.data().compute_normals();
  viewer.core().align_camera_center(V, F);
  Eigen::MatrixXd N;
  igl::per_vertex_normals(V, F, N);
  Eigen::MatrixXd C = N.rowwise().normalized().array() * 0.5 + 0.5;
  viewer.data().set_colors(C);
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

  if (!igl::readOBJ(args.input, OV, OF)) {
    std::cerr << "Failed to read the mesh" << std::endl;
    return 0;
  }

  V = OV;
  F = OF;
  if (OF.cols() == 4) {
    std::cout << "It is a tet mesh" << std::endl;
  } else if(OF.cols() == 3){
    std::cout << "It is a triangle mesh" << std::endl;
  }

  bool activate_parth = args.activate_parth;
  int TOTAL_TEST = 10;
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
        PARTH_SOLVER::LinSysSolver::create(PARTH_SOLVER::LinSysSolverType::MKL);
  } else {
    std::cerr << "UNKNOWN linear solver" << std::endl;
    return 0;
  }

  if (solver == nullptr) {
    std::cerr << "Solver is not created" << std::endl;
    return 0;
  }
  // Solver Config
  solver->setSimulationDIM(1);
  solver->setParthActivation(activate_parth);
  std::vector<int> new_to_old_map;
  Smoothing(new_to_old_map, data_name, true, solver);
  // Main loop for experiment -> Parth Timing
  for (int test_cnt = 0; test_cnt < TOTAL_TEST; test_cnt++) {
    std::cout << "----------------- Test number: " << test_cnt
              << " -----------------" << std::endl;
    new_to_old_map.clear();
    remesh(F, V, args.scale, new_to_old_map);
    // Solving the laplacian
    Smoothing(new_to_old_map, data_name, true, solver);
  }


  if(show_vis){
    igl::opengl::glfw::Viewer viewer;

    int frame = 0;
    reset();
    solver->resetSolver();
    solver->resetTotalTime();


    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
      if (frame == TOTAL_TEST) {
        viewer.core().is_animating = false;
        return true;
      } else {
        viewer.core().is_animating = true;
      }
      if (viewer.core().is_animating) {
        UpdateViewer(viewer);
        std::cout << "Frame: " << frame << std::endl;
      }
      frame++;
      return false;
    };
    bool init_gif = false;
    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
      if (viewer.core().is_animating) {
        std::vector<int> new_to_old_map;
        remesh(F, V, args.scale, new_to_old_map);
        Smoothing(new_to_old_map, data_name, false, solver);
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
  }

  delete solver;
  return 0;
}