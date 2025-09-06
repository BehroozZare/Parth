//
// Created by behrooz zare on 2024-04-06.
//
#include "Parth.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cholmod.h>
#include <igl/cotmatrix.h>
#include <igl/false_barycentric_subdivision.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/setdiff.h>
#include <igl/slice_mask.h>
#include <igl/unique.h>
#include <igl/upsample.h>
#include <iostream>
#include <omp.h>


Eigen::SparseMatrix<double>
computeMeshFromHessian(Eigen::SparseMatrix<double> &A, int DIM) {
  Eigen::SparseMatrix<double> mesh_csc;
  assert(A.rows() % DIM == 0);
  mesh_csc.resize(A.rows() / DIM, A.cols() / DIM);
  std::vector<Eigen::Triplet<double>> coefficients;
  int N = A.rows();
  int *Ap = A.outerIndexPtr();
  int *Ai = A.innerIndexPtr();

  for (int c = 0; c < N; c += DIM) {
    assert((Ap[c + 1] - Ap[c]) % DIM == 0);
    for (int r_ptr = Ap[c]; r_ptr < Ap[c + 1]; r_ptr += DIM) {
      int r = Ai[r_ptr];
      int mesh_c = c / DIM;
      int mesh_r = r / DIM;
      if (mesh_c != mesh_r) {
        coefficients.emplace_back(mesh_c, mesh_r, 1);
        coefficients.emplace_back(mesh_r, mesh_c, 1);
      }
    }
  }

  mesh_csc.setFromTriplets(coefficients.begin(), coefficients.end());
  //  assert((A.nonZeros() * 2 - A.rows()) / (DIM * DIM) - mesh_csc.rows() ==
  //         mesh_csc.nonZeros());
  return mesh_csc;
}

class AccelerateWrapper { // The class
public:                // Access specifier
  double load_time;
  double analyze_time;
  double factor_time;
  double solve_time;
  double residual;
  double L_NNZ;
  double parth_time;

  PARTH::Parth parth;
  std::vector<int> perm;
  bool parth_activation = false;

  cholmod_common cm;
  cholmod_sparse *A;
  cholmod_factor *L;
  cholmod_dense *b, *solution;

  cholmod_dense *x_solve;

  void *Ai, *Ap, *Ax, *bx;
  int N;

  ~AccelerateWrapper() {
    this->cholmod_clean_memory();

    cholmod_finish(&cm);
  }

    AccelerateWrapper(bool activate_immobilizer = false) {
    cholmod_start(&cm);
    A = NULL;
    L = NULL;
    b = NULL;
    solution = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
    bx = NULL;

    this->parth_activation = activate_immobilizer;

    if (parth_activation) {
      std::cout << "Activating Immobilizer" << std::endl;
    } else {
      std::cout << "Deactivating Immobilizer" << std::endl;
    }

    parth.setReorderingType(PARTH::ReorderingType::METIS);
    parth.setVerbose(true);
    parth.setNDLevels(6);
    parth.setNumberOfCores(10);
  }

  void cholmod_clean_memory() {
    if (A) {
      A->i = Ai;
      A->p = Ap;
      A->x = Ax;
      cholmod_free_sparse(&A, &cm);
    }

    cholmod_free_factor(&L, &cm);

    if (b) {
      b->x = bx;
      cholmod_free_dense(&b, &cm);
    }

    if (x_solve) {
      cholmod_free_dense(&x_solve, &cm);
    }
  }

  void printTiming() {
    parth.printTiming();
    double total_time = load_time + analyze_time + factor_time + solve_time;
    std::cout << "The analysis time is: " << (analyze_time / total_time) * 100
              << "%" << std::endl;
    std::cout << "The factorization time is: "
              << (factor_time / total_time) * 100 << "%" << std::endl;
    std::cout << "The solve time is: " << (solve_time / total_time) * 100 << "%"
              << std::endl;
    std::cout << "Total Solve time: " << total_time << " seconds" << std::endl;
  }

  void setMatrix(int *p, int *i, double *x, int A_N, int NNZ, int *Mp, int *Mi,
                 int M_N, std::vector<int> &new_to_old_map) {

    assert(p[A_N] == NNZ);
    this->N = A_N;

    assert(new_to_old_map.size() == M_N);
    if (parth_activation) {
      parth.setMeshPointers(M_N, Mp, Mi, new_to_old_map);
      std::cout << "Using Parth ..." << std::endl;
    } else {
      parth.setMeshPointers(M_N, Mp, Mi);
    }

    this->cholmod_clean_memory();

    if (!A) {
      A = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
      this->Ap = A->p;
      this->Ax = A->x;
      this->Ai = A->i;
      // -1: upper right part will be ignored during computation
    }

    A->p = p;
    A->i = i;
    A->x = x;
  }

  void analyze_pattern(void) {
    analyze_time = omp_get_wtime();
    parth_time = omp_get_wtime();
    if (parth_activation) {
      parth.computePermutation(perm, 1);
    }
    parth_time = omp_get_wtime() - parth_time;

    cholmod_free_factor(&L, &cm);

    cm.supernodal = CHOLMOD_SUPERNODAL;
    if (parth_activation) {
      cm.nmethods = 1;
      cm.method[0].ordering = CHOLMOD_GIVEN;
      L = cholmod_analyze_p(A, perm.data(), NULL, 0, &cm);
      std::cout << "The method with Immobilizer is: " << cm.selected
                << std::endl;
    } else {
      cm.nmethods = 1;
      cm.method[0].ordering = CHOLMOD_METIS;
      L = cholmod_analyze(A, &cm);
      if (cm.selected == 1) {
        std::cout << "The method is AMD." << std::endl;
      } else if (cm.selected == 2) {
        std::cout << "The method is METIS." << std::endl;
      } else if (cm.selected == 0) {
        std::cout << "The method is UserDefined." << std::endl;
      } else {
        std::cout << "The method " << cm.selected << " is unknown" << std::endl;
      }
    }

    L_NNZ = cm.lnz * 2 - N;

    analyze_time = omp_get_wtime() - analyze_time;
    if (L == nullptr) {
      std::cerr << "ERROR during symbolic factorization:" << std::endl;
    }
  }

  bool factorize(void) {
    factor_time = omp_get_wtime();
    cholmod_factorize(A, L, &cm);
    factor_time = omp_get_wtime() - factor_time;
    if (cm.status == CHOLMOD_NOT_POSDEF) {
      std::cerr << "ERROR during numerical factorization - code: " << std::endl;
      return false;
    }
    return true; // TODO:CHECK FOR SPD FLAGS LATER
  }

  void solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
    /* -------------------------------------------------------------------- */
    /* .. Back substitution and iterative refinement. */
    /* -------------------------------------------------------------------- */
    if (!b) {
      b = cholmod_allocate_dense(N, 1, N, CHOLMOD_REAL, &cm);
      bx = b->x;
    }
    b->x = rhs.data();

    if (x_solve) {
      cholmod_free_dense(&x_solve, &cm);
    }

    solve_time = omp_get_wtime();
    x_solve = cholmod_solve(CHOLMOD_A, L, b, &cm);
    solve_time = omp_get_wtime() - solve_time;

    result.conservativeResize(rhs.size());
    memcpy(result.data(), x_solve->x, result.size() * sizeof(result[0]));
  }
};

Eigen::MatrixXi OF, F;
Eigen::MatrixXd OV, V;
Eigen::MatrixXd B;
AccelerateWrapper Accelerate_Parth(false);
Eigen::VectorXi in_prev;
Eigen::VectorXi in_prev_inv;

int step = 0;

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key,
              int modifier) {
  if (key == '0') {
    V = OV;
    F = OF;
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    CHOLMOD_Immobilizer.parth_activation =
        !CHOLMOD_Immobilizer.parth_activation;
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
    viewer.core().trackball_angle = Eigen::Quaternionf(
        Eigen::AngleAxisf(M_PI_4 / 2.0, Eigen::Vector3f::UnitX()));
>>>>>>> f5a1412a376d1b5a40566a0fe5fdceac8ba3ea22
    return false;
  }
  // Chose a random point in the mesh
  if (key >= '1' && key <= '9') {
    int t = int((key - '1') + 1);
    std::cout << "**************** Step: " << step << std::endl;
    double total_time = omp_get_wtime();
    //------------------- Randomly select t faces -------------------
    std::cout << "Up sampling " << t << " faces" << std::endl;
    Eigen::VectorXi I(t);
    for (int i = 0; i < t; i++) {
      I(i) = rand() % F.rows();
    }
    std::vector<bool> index_chosen(F.rows(), false);
    for (int i = 0; i < I.rows(); i++) {
      index_chosen[I(i)] = true;
    }

    // Compute a local face set and vertices set
    Eigen::MatrixXi F_sub;
    Eigen::MatrixXd V_sub;

    // Assign the local face set and vertices set
    igl::slice(F, I, 1, F_sub);
    std::vector<int> local_to_global_map;
    std::vector<int> global_to_local_map;
    // Compute the set of vertices used in the F_sub by iterating over each face
    for (int r = 0; r < F_sub.rows(); r++) {
      // Add each vertex of the face to the set of vertices
      for (int c = 0; c < F_sub.cols(); c++) {
        local_to_global_map.push_back(F_sub(r, c));
      }
    }

    // Delete duplicated vertices and sort the vertices_ids
    std::sort(local_to_global_map.begin(), local_to_global_map.end());
    local_to_global_map.erase(
        std::unique(local_to_global_map.begin(), local_to_global_map.end()),
        local_to_global_map.end());

    // Create V_sub using vertices_ids
    V_sub.resize(local_to_global_map.size(), V.cols());
    for (int i = 0; i < local_to_global_map.size(); i++) {
      V_sub.row(i) = V.row(local_to_global_map[i]);
    }

    // Compute global to local ids
    global_to_local_map.resize(V.rows());
    for (int i = 0; i < local_to_global_map.size(); i++) {
      global_to_local_map[local_to_global_map[i]] = i;
    }

    // Map the F_sub based on local vertices
    for (int r = 0; r < F_sub.rows(); r++) {
      for (int c = 0; c < F_sub.cols(); c++) {
        F_sub(r, c) = global_to_local_map[F_sub(r, c)];
      }
    }

    // ------------------- Up sample the selected faces -------------------
    int prev_num_nodes = V_sub.rows();
    int prev_num_faces = F_sub.rows();
    igl::upsample(Eigen::MatrixXd(V_sub), Eigen::MatrixXi(F_sub), V_sub, F_sub);

    // ------------------- Integrate the up sampled Vertices and faces
    Eigen::MatrixXd V_up;
    Eigen::MatrixXi F_up;

    // ------ Integrate vertices
    V_up.resize(V.rows() + V_sub.rows() - prev_num_nodes, V.cols());
    for (int i = 0; i < V.rows(); i++) {
      V_up.row(i) = V.row(i);
    }

    for (int i = prev_num_nodes; i < V_sub.rows(); i++) {
      V_up.row(V.rows() + i - prev_num_nodes) = V_sub.row(i);
    }

    // ------ Integrate faces

    // node mapping
    local_to_global_map.resize(V_sub.rows());
    for (int i = 0; i < V_sub.rows() - prev_num_nodes; i++) {
      local_to_global_map[i + prev_num_nodes] = V.rows() + i;
      assert(local_to_global_map[i] < V_up.rows() && "Index out of bound");
    }

    // Map the faces to their global vertices
    F_up.resize(F.rows() + F_sub.rows() - prev_num_faces, F.cols());
    int cnt = 0;
    for (int i = 0; i < F.rows(); i++) {
      if (!index_chosen[i]) {
        F_up.row(cnt++) = F.row(i);
      }
    }

    for (int j = 0; j < F_sub.rows(); j++) {
      for (int i = 0; i < F_sub.cols(); i++) {
        F_up.row(cnt)(i) = local_to_global_map[F_sub.row(j)(i)];
        if (F_up.row(cnt)(i) >= V_up.rows()) {
          std::cout << F_up.row(cnt)(i) << " >= " << V_up.rows()
                    << " - Index out of bound" << std::endl;
        }
        assert(F_up.row(cnt)(i) < V_up.rows() && "Index out of bound");
      }
      cnt++;
    }

    //------------------- Solve the PDE -------------------
    std::cout << "Solving Laplacian PDE" << std::endl;
    //    // Find boundary_facets edges
    //    Eigen::MatrixXi E;
    //    igl::boundary_facets(F_up, E);
    //    // Find boundary vertices
    //    Eigen::VectorXi b, IA, IC;
    //    igl::unique(E, b, IA, IC);
    //    // List of all vertex indices
    //    Eigen::VectorXi all, in;
    //    igl::colon<int>(0, V_up.rows() - 1, all);
    //    // List of interior indices
    //    igl::setdiff(all, b, in, IA);

    // Compute the maximum and minimum x and y
    double max_x = V_up.col(0).maxCoeff();
    double min_x = V_up.col(0).minCoeff();
    double max_y = V_up.col(1).maxCoeff();
    double min_y = V_up.col(1).minCoeff();

    // Compute the boundary nodes based on whether the nodes are within 1.1 *
    // min or outside of 0.9 * max
    std::vector<int> in_tmp;
    std::vector<int> b_tmp;
    for (int i = 0; i < V_up.rows(); i++) {
      if (V_up(i, 0) < min_x + 0.1 || V_up(i, 0) > 0.9 * max_x ||
          V_up(i, 1) < min_y + 0.1 || V_up(i, 1) > 0.9 * max_y) {
        in_tmp.emplace_back(i);
      } else {
        b_tmp.emplace_back(i);
      }
    }

    // Copy in_tmp and b_tmp to Eigen::VectorXi in and b
    Eigen::VectorXi in, b;
    in.resize(in_tmp.size());
    b.resize(b_tmp.size());
    for (int i = 0; i < in_tmp.size(); i++) {
      in(i) = in_tmp[i];
    }
    for (int i = 0; i < b_tmp.size(); i++) {
      b(i) = b_tmp[i];
    }
    // Construct and slice up Laplacian
    Eigen::SparseMatrix<double> L, L_in_in, L_in_b;
    igl::cotmatrix(V_up, F_up, L);
    igl::slice(L, in, in, L_in_in);
    igl::slice(L, in, b, L_in_b);
    std::cout << "Number of inner dofs are: " << in.rows() << std::endl;
    // Dirichlet boundary conditions from z-coordinate
    Eigen::VectorXd Z = V_up.col(2);
    Eigen::VectorXd bc = Z(b);

    // Computing the mapper:
    std::vector<int> new_to_old_map;
    for (int i = 0; i < V.rows(); i++) {
      new_to_old_map.push_back(i);
    }
    for (int j = V.rows(); j < V_up.rows(); j++) {
      new_to_old_map.push_back(-1);
    }

    // Computing the mapping for Parth (due to boundary condition)
    std::vector<int> parth_new_to_old_map(in.rows());
    Eigen::VectorXi in_inv(V_up.rows());
    in_inv.setOnes();
    in_inv = -in_inv;
    // computing in_prev_inv
    for (int i = 0; i < in.rows(); i++) {
      in_inv(in(i)) = i;
    }

    for (int parth_dof_idx = 0; parth_dof_idx < in.rows(); parth_dof_idx++) {
      int curr_dof = in(parth_dof_idx);
      int prev_dof = new_to_old_map[curr_dof];

      // If it is a newly added
      if (prev_dof == -1) {
        parth_new_to_old_map[parth_dof_idx] = -1;
        continue;
      }

      // If it is the first time that the function is called
      if (in_prev_inv.rows() == 0) {
        parth_new_to_old_map[parth_dof_idx] = parth_dof_idx;
        continue;
      }

      int curr_dof_pos = in_inv(curr_dof);
      int prev_dof_pos = in_prev_inv(prev_dof);
      assert(parth_dof_idx == curr_dof_pos);
      // If it existed but was not selected before
      if (prev_dof_pos == -1) {
        parth_new_to_old_map[parth_dof_idx] = -1;
        continue;
      }

      // If it existed and selected before
      parth_new_to_old_map[parth_dof_idx] = prev_dof_pos;
    }

    Eigen::SparseMatrix<double> mesh = computeMeshFromHessian(L_in_in, 1);

    // Solve PDE
    L_in_in = -L_in_in;
    CHOLMOD_Immobilizer.setMatrix(
        L_in_in.outerIndexPtr(), L_in_in.innerIndexPtr(), L_in_in.valuePtr(),
        L_in_in.rows(), L_in_in.nonZeros(), mesh.outerIndexPtr(),
        mesh.innerIndexPtr(), mesh.rows(), parth_new_to_old_map);

    CHOLMOD_Immobilizer.analyze_pattern();
    CHOLMOD_Immobilizer.factorize();
    Eigen::VectorXd rhs = L_in_b * bc;
    Eigen::VectorXd sol;
    CHOLMOD_Immobilizer.solve(rhs, sol);
    // slice into solution
    Z(in) = sol;
    // ------------------- Color the faces -------------------
    std::cout << "Coloring the faces" << std::endl;
    //        std::vector<int> I_extend_tmp;
    //        for (int i = 0; i < I.size(); i++) {
    //            I_extend_tmp.push_back(I(i));
    //        }
    //
    //        for(int j = F.rows() - prev_num_faces; j < F_up.rows(); j++){
    //            I_extend_tmp.push_back(j);
    //        }
    //        // Copy I_extend_tmp to Eigen::VectorXi I_extend
    //        Eigen::VectorXi I_extend;
    //        I_extend.resize(I_extend_tmp.size());
    //        for (int i = 0; i < I_extend_tmp.size(); i++) {
    //            I_extend(i) = I_extend_tmp[i];
    //        }
    //
    //        // default green for all faces
    //        Eigen::MatrixXd C =
    //                Eigen::RowVector3d(0.4, 0.8, 0.3).replicate(F_up.rows(),
    //                1);
    //            // Red for each in K
    //        Eigen::MatrixXd R =
    //                Eigen::RowVector3d(1.0, 0.3,
    //                0.3).replicate(I_extend.rows(), 1);
    //
    //        C(I_extend, Eigen::indexing::all) = R;\

    total_time = omp_get_wtime() - total_time;
    CHOLMOD_Immobilizer.printTiming();
    std::cout << "Total Process time for Step: " << step << " is " << total_time
              << " seconds" << std::endl;

    viewer.data().clear();
    // Plot the mesh with pseudocolors
    viewer.data().set_mesh(V_up, F_up);
    //        viewer.data().show_lines = false;
    viewer.data().set_data(Z);
    viewer.data().set_face_based(true);
    // Setup viewer
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
    viewer.core().trackball_angle = Eigen::Quaternionf(
        Eigen::AngleAxisf(M_PI_4 / 2.0, Eigen::Vector3f::UnitX()));
    V = V_up;
    F = F_up;
    in_prev = in;
    in_prev_inv = in_inv;

    step++;
    std::cout << "**************** Step " << step << " completed" << std::endl;
  }
  return false;
}

int main(int argc, char *argv[]) {

  igl::read_triangle_mesh("/Users/behroozzare/Desktop/Graphics/ParthSolverDev/data/jello_hi.obj",
                          OV, OF);
//    igl::read_triangle_mesh("/Users/behroozzare/Desktop/Graphics/ParthSolverDev/data/toy/bar_2d.obj",
//                            OV, OF);
    V = OV;
    F = OF;

  igl::barycenter(OV, OF, B);

  std::cout << R"(Usage:
   0  Restore Original mesh
   1  Up sample and solve PDE
  )";
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_face_based(true);

  viewer.callback_key_down = &key_down;
  key_down(viewer, '0', 0);
  viewer.launch();

  return 0;
}