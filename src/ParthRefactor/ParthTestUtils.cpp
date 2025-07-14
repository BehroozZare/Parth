//
// Created by behrooz zare on 2024-04-08.
//
#include "ParthTestUtils.h"
#include "cholmod.h"
#include "omp.h"
#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

namespace PARTH {

int ParthTestUtils::getLNNZ(int NNZ, int N, int *Ap, int *Ai, double *Ax,
                            int *perm) {
  cholmod_common cm;
  cholmod_sparse *A;
  cholmod_factor *L;

  cholmod_start(&cm);
  A = NULL;
  L = NULL;
  void *Ai_tmp, *Ap_tmp, *Ax_tmp;

  if (!A) {
    A = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
    Ap_tmp = A->p;
    Ax_tmp = A->x;
    Ai_tmp = A->i;
    // -1: upper right part will be ignored during computation
  }

  A->p = Ap;
  A->i = Ai;
  A->x = Ax;

  cm.nmethods = 1;
  cm.method[0].ordering = CHOLMOD_GIVEN;
  L = cholmod_analyze_p(A, perm, NULL, 0, &cm);

  int L_NNZ = cm.lnz * 2 - N;

  if (L == nullptr) {
    std::cerr << "ERROR during symbolic factorization:" << std::endl;
  }

  if (A) {
    A->i = Ai_tmp;
    A->p = Ap_tmp;
    A->x = Ax_tmp;
    cholmod_free_sparse(&A, &cm);
  }

  cholmod_free_factor(&L, &cm);
  cholmod_finish(&cm);

  return L_NNZ;
}

int ParthTestUtils::getLNNZ_AMD(int NNZ, int N, int *Ap, int *Ai, double *Ax) {
  cholmod_common cm;
  cholmod_sparse *A;
  cholmod_factor *L;

  cholmod_start(&cm);
  A = NULL;
  L = NULL;
  void *Ai_tmp, *Ap_tmp, *Ax_tmp;

  if (!A) {
    A = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
    Ap_tmp = A->p;
    Ax_tmp = A->x;
    Ai_tmp = A->i;
    // -1: upper right part will be ignored during computation
  }

  A->p = Ap;
  A->i = Ai;
  A->x = Ax;

  cm.nmethods = 1;
  cm.method[0].ordering = CHOLMOD_AMD;
  L = cholmod_analyze(A, &cm);

  int L_NNZ = cm.lnz * 2 - N;

  if (L == nullptr) {
    std::cerr << "ERROR during symbolic factorization:" << std::endl;
  }

  if (A) {
    A->i = Ai_tmp;
    A->p = Ap_tmp;
    A->x = Ax_tmp;
    cholmod_free_sparse(&A, &cm);
  }

  cholmod_free_factor(&L, &cm);
  cholmod_finish(&cm);

  return L_NNZ;
}

int ParthTestUtils::getLNNZ_METIS(int NNZ, int N, int *Ap, int *Ai,
                                  double *Ax) {
  cholmod_common cm;
  cholmod_sparse *A;
  cholmod_factor *L;

  cholmod_start(&cm);
  A = NULL;
  L = NULL;
  void *Ai_tmp, *Ap_tmp, *Ax_tmp;

  if (!A) {
    A = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
    Ap_tmp = A->p;
    Ax_tmp = A->x;
    Ai_tmp = A->i;
    // -1: upper right part will be ignored during computation
  }

  A->p = Ap;
  A->i = Ai;
  A->x = Ax;

  cm.nmethods = 1;
  cm.method[0].ordering = CHOLMOD_METIS;
  L = cholmod_analyze(A, &cm);

  int L_NNZ = cm.lnz * 2 - N;

  if (L == nullptr) {
    std::cerr << "ERROR during symbolic factorization:" << std::endl;
  }

  if (A) {
    A->i = Ai_tmp;
    A->p = Ap_tmp;
    A->x = Ax_tmp;
    cholmod_free_sparse(&A, &cm);
  }

  cholmod_free_factor(&L, &cm);
  cholmod_finish(&cm);

  return L_NNZ;
}

void ParthTestUtils::testPermutationQuality(std::string hessian_name,
                                            std::string mesh_name) {
  //************** Load the Mesh  *************
  Eigen::SparseMatrix<double> mesh_csc;
  if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
    std::cerr << "File " << mesh_name << " is not found" << std::endl;
  }

  Eigen::SparseMatrix<double> lower_A_csc;
  if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
    std::cerr << "File " << hessian_name << " is not found" << std::endl;
  }

  int M_n = mesh_csc.rows();
  int *Mp = mesh_csc.outerIndexPtr();
  int *Mi = mesh_csc.innerIndexPtr();

  int A_n = lower_A_csc.rows();
  int A_nnz = lower_A_csc.nonZeros();
  int *Ap = lower_A_csc.outerIndexPtr();
  int *Ai = lower_A_csc.innerIndexPtr();
  double *Ax = lower_A_csc.valuePtr();

  std::vector<int> parth_perm;

  PARTH::Parth parth;
  parth.setReorderingType(ReorderingType::METIS);
  parth.setNDLevels(6);
  parth.setNumberOfCores(4);
  parth.setVerbose(true);
  parth.setMeshPointers(M_n, Mp, Mi);
  parth_perm.resize(A_n);
  assert(parth_perm.size() == A_n);

  std::cout << "Computing NNZ for Parth permutation" << std::endl;
  double parth_metis_time = omp_get_wtime();
  parth.computePermutation(parth_perm);
  int parth_nnz = getLNNZ(A_nnz, A_n, Ap, Ai, Ax, parth_perm.data());
  parth_metis_time = omp_get_wtime() - parth_metis_time;

  std::cout << "Computing NNZ for METIS permutation" << std::endl;
  double metis_time = omp_get_wtime();
  int metis_nnz = getLNNZ_METIS(A_nnz, A_n, Ap, Ai, Ax);
  metis_time = omp_get_wtime() - metis_time;

  double diff = (double)(metis_nnz - parth_nnz) / (double)metis_nnz;
  std::cout
      << "Difference between METIS and PARTH -> the more is better for parth: "
      << diff << std::endl;
  std::cout << "The Parth + Metis time Vs. Metis " << parth_metis_time << " - "
            << metis_time << std::endl;

  //==============================================================================================================
  parth.clearParth();
  parth.setReorderingType(ReorderingType::AMD);
  parth.setNDLevels(6);
  parth.setNumberOfCores(4);
  parth.setVerbose(true);
  parth.setMeshPointers(M_n, Mp, Mi);

  double parth_amd_time = omp_get_wtime();
  parth.computePermutation(parth_perm);
  parth_nnz = getLNNZ(A_nnz, A_n, Ap, Ai, Ax, parth_perm.data());
  parth_amd_time = omp_get_wtime() - parth_amd_time;

  std::cout << "Computing NNZ for AMD permutation" << std::endl;
  double AMD_time = omp_get_wtime();
  int AMD_nnz = getLNNZ_AMD(A_nnz, A_n, Ap, Ai, Ax);
  AMD_time = omp_get_wtime() - AMD_time;

  diff = (double)(AMD_nnz - parth_nnz) / (double)AMD_nnz;
  std::cout
      << "Difference between AMD and PARTH -> the more is better for parth: "
      << diff << std::endl;
  std::cout << "The Parth + AMD time Vs. AMD " << parth_amd_time << " - "
            << AMD_time << std::endl;

  diff = (double)(metis_nnz - parth_nnz) / (double)metis_nnz;
  std::cout << "Difference between PARTH + AMD and METIS -> the more is better "
               "for parth: "
            << diff << std::endl;
}

double ParthTestUtils::ReuseOfOne(std::string hessian_name,
                                  std::string mesh_name) {
  //************** Load the Mesh  *************
  Eigen::SparseMatrix<double> mesh_csc;
  if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
    std::cerr << "File " << mesh_name << " is not found" << std::endl;
  }

  Eigen::SparseMatrix<double> lower_A_csc;
  if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
    std::cerr << "File " << hessian_name << " is not found" << std::endl;
  }

  //************** setup Parth *************
  PARTH::Parth parth;
  parth.setReorderingType(PARTH::ReorderingType::METIS);
  parth.setVerbose(true);
  parth.setNDLevels(2);
  parth.setNumberOfCores(4);
  //************ Check the adding and deleting edge functionality ***********
  // Reuse of 1 when separator and left and right sub-meshes are connected
  std::vector<int> parth_perm(lower_A_csc.rows());
  parth.setMeshPointers(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                        mesh_csc.innerIndexPtr());
  parth.computePermutation(parth_perm);
  assert(parth.hmd.num_HMD_nodes == 7);

  // Add the edge to the mesh
  std::vector<int> Mp_new;
  std::vector<int> Mi_new;
  std::vector<std::pair<int, int>> edge_set;
  std::vector<std::pair<int, int>> parth_edge_set;
  parth_edge_set.push_back({0, 1});
  this->createEdges(parth, parth_edge_set, 1, 2, edge_set);
  this->addEdgeToMesh(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                      mesh_csc.innerIndexPtr(), Mp_new, Mi_new, edge_set);

  parth.setMeshPointers(mesh_csc.rows(), Mp_new.data(), Mi_new.data());
  parth.computePermutation(parth_perm);
  return parth.getReuse();
}

double ParthTestUtils::ReuseOfZero(std::string hessian_name,
                                   std::string mesh_name) {
  //************** Load the Mesh  *************
  Eigen::SparseMatrix<double> mesh_csc;
  if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
    std::cerr << "File " << mesh_name << " is not found" << std::endl;
  }

  Eigen::SparseMatrix<double> lower_A_csc;
  if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
    std::cerr << "File " << hessian_name << " is not found" << std::endl;
  }

  //************** setup Parth *************
  PARTH::Parth parth;
  parth.setReorderingType(PARTH::ReorderingType::METIS);
  parth.setVerbose(true);
  parth.setNDLevels(2);
  parth.setNumberOfCores(4);
  //************ Check the adding and deleting edge functionality ***********
  // Reuse of 1 when separator and left and right sub-meshes are connected
  std::vector<int> parth_perm(lower_A_csc.rows());
  parth.setMeshPointers(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                        mesh_csc.innerIndexPtr());
  parth.computePermutation(parth_perm);
  assert(parth.hmd.num_HMD_nodes == 7);

  // Add the edge to the mesh
  std::vector<int> Mp_new;
  std::vector<int> Mi_new;
  std::vector<std::pair<int, int>> edge_set;
  std::vector<std::pair<int, int>> parth_edge_set;
  parth_edge_set.push_back({1, 2});
  this->createEdges(parth, parth_edge_set, 1, 2, edge_set);
  this->addEdgeToMesh(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                      mesh_csc.innerIndexPtr(), Mp_new, Mi_new, edge_set);

  parth.setMeshPointers(mesh_csc.rows(), Mp_new.data(), Mi_new.data());
  parth.computePermutation(parth_perm);
  return parth.getReuse();
}

bool ParthTestUtils::DirtyNodeDetectionCheck(std::string hessian_name,
                                             std::string mesh_name) {
  //************** Load the Mesh  *************
  Eigen::SparseMatrix<double> mesh_csc;
  if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
    std::cerr << "File " << mesh_name << " is not found" << std::endl;
  }

  Eigen::SparseMatrix<double> lower_A_csc;
  if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
    std::cerr << "File " << hessian_name << " is not found" << std::endl;
  }

  //************** setup Parth *************
  PARTH::Parth parth;
  parth.setReorderingType(PARTH::ReorderingType::METIS);
  parth.setVerbose(true);
  parth.setNDLevels(2);
  parth.setNumberOfCores(4);
  //************ Check the adding and deleting edge functionality ***********
  // Reuse of 1 when separator and left and right sub-meshes are connected
  std::vector<int> parth_perm(lower_A_csc.rows());
  parth.setMeshPointers(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                        mesh_csc.innerIndexPtr());
  parth.computePermutation(parth_perm);
  assert(parth.hmd.num_HMD_nodes == 7);

  // Add the edge to the mesh
  std::vector<int> Mp_new;
  std::vector<int> Mi_new;
  std::vector<std::pair<int, int>> edge_set;
  std::vector<std::pair<int, int>> parth_edge_set;
  parth_edge_set.push_back({3, 4});
  parth_edge_set.push_back({3, 3});
  parth_edge_set.push_back({2, 2});
  parth_edge_set.push_back({4, 4});
  parth_edge_set.push_back({5, 5});
  parth_edge_set.push_back({6, 6});
  parth_edge_set.push_back({0, 2});
  parth_edge_set.push_back({0, 5});
  parth_edge_set.push_back({0, 3});
  this->createEdges(parth, parth_edge_set, 1, 100, edge_set);
  this->addEdgeToMesh(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                      mesh_csc.innerIndexPtr(), Mp_new, Mi_new, edge_set);

  parth.setMeshPointers(mesh_csc.rows(), Mp_new.data(), Mi_new.data());
  parth.computePermutation(parth_perm);
  std::vector<int> expected_dirty_fine_grain = {2, 5, 6};
  std::vector<int> expected_coarse_grain = {1};

  // check whether dirty_fine_Grains are only consist of 5 and 6
  if (parth.integrator.dirty_fine_HMD_nodes_saved.size() !=
      expected_dirty_fine_grain.size()) {
    return false;
  } else {
    // find each element dirty_fine_HMD_nodes_saved in the
    // expected_dirty_fine_grain
    for (int i = 0; i < parth.integrator.dirty_fine_HMD_nodes_saved.size();
         i++) {
      if (std::find(expected_dirty_fine_grain.begin(),
                    expected_dirty_fine_grain.end(),
                    parth.integrator.dirty_fine_HMD_nodes_saved[i]) ==
          expected_dirty_fine_grain.end()) {
        return false;
      }
    }
  }
  // Do the same procedure for coarse grain
  if (parth.integrator.dirty_coarse_HMD_nodes_saved.size() !=
      expected_coarse_grain.size()) {
    return false;
  } else {
    for (int i = 0; i < parth.integrator.dirty_coarse_HMD_nodes_saved.size();
         i++) {
      if (std::find(expected_coarse_grain.begin(), expected_coarse_grain.end(),
                    parth.integrator.dirty_coarse_HMD_nodes_saved[i]) ==
          expected_coarse_grain.end()) {
        return false;
      }
    }
  }
  std::cout << "Parth reuse is: " << parth.getReuse() << " and it handled "
            << parth.getNumChanges() << " Changes in the Mesh " << std::endl;
  return true;
}

void ParthTestUtils::testAddingDeletingEdges(std::string hessian_name,
                                             std::string mesh_name) {

  // Reuse of 0 when left and right sub-meshes of root separator are connected
  // Add edges with zero reuse
  double reuse_one = this->ReuseOfOne(hessian_name, mesh_name);
  if (reuse_one != 1.0) {
    std::cerr << "The reuse in this test should be one but it is: " << reuse_one
              << std::endl;
    assert(false);
  }

  double reuse_zero = this->ReuseOfZero(hessian_name, mesh_name);
  if (reuse_zero != 0) {
    std::cerr << "The reuse in this test should be zero but it is: "
              << reuse_zero << std::endl;
    assert(false);
  }

  if (!this->DirtyNodeDetectionCheck(hessian_name, mesh_name)) {
    std::cerr << "The dirty node detection is not working properly"
              << std::endl;
    assert(false);
  }
}

void ParthTestUtils::testDeletingDOFs(std::string hessian_name,
                                      std::string mesh_name) {
  //************** Load the Mesh  *************
  Eigen::SparseMatrix<double> mesh_csc;
  if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
    std::cerr << "File " << mesh_name << " is not found" << std::endl;
  }

  Eigen::SparseMatrix<double> lower_A_csc;
  if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
    std::cerr << "File " << hessian_name << " is not found" << std::endl;
  }

  //************** setup Parth *************
  PARTH::Parth parth;
  parth.setReorderingType(PARTH::ReorderingType::METIS);
  parth.setVerbose(true);
  parth.setNDLevels(2);
  parth.setNumberOfCores(4);
  //************ Check the adding and deleting edge functionality ***********
  // Reuse of 1 when separator and left and right sub-meshes are connected
  std::vector<int> parth_perm(lower_A_csc.rows());
  parth.setMeshPointers(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                        mesh_csc.innerIndexPtr());
  parth.computePermutation(parth_perm);
  assert(parth.hmd.num_HMD_nodes == 7);

  // Add the edge to the mesh
  std::vector<int> Mp_new;
  std::vector<int> Mi_new;
  std::vector<double> Mx_test;
  std::vector<int> HMD_nodes_to_delete_dof = {1, 2, 3, 4, 5, 6};
  std::vector<int> dof_mapper;
  this->deleteDOF(parth, mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                  mesh_csc.innerIndexPtr(), 1, 2, Mp_new, Mi_new,
                  HMD_nodes_to_delete_dof, dof_mapper);
  parth.setMeshPointers(Mp_new.size() - 1, Mp_new.data(), Mi_new.data());
  parth.setNewToOldDOFMap(dof_mapper);
  parth.computePermutation(parth_perm);
  assert(parth.getReuse() == 1);
  Mx_test.resize(Mi_new.size());
  // Add random values for Mx_test
  for (int i = 0; i < Mi_new.size(); i++) {
    Mx_test[i] = rand() % 100;
  }
  // Measure the time
  double parth_time = omp_get_wtime();
  auto parth_nnz =
      this->getLNNZ(Mi_new.size(), Mp_new.size() - 1, Mp_new.data(),
                    Mi_new.data(), Mx_test.data(), parth.mesh_perm.data());
  parth_time = omp_get_wtime() - parth_time;

  // Measure the time for METIS
  double metis_time = omp_get_wtime();
  auto metis_nnz =
      this->getLNNZ_METIS(Mi_new.size(), Mp_new.size() - 1, Mp_new.data(),
                          Mi_new.data(), Mx_test.data());
  metis_time = omp_get_wtime() - metis_time;
  // Print the difference in quality
  std::cout
      << "Difference between METIS and PARTH -> the more is better for parth: "
      << (double)(metis_nnz - parth_nnz) / (double)metis_nnz << std::endl;
  // Print the time
  std::cout << "The Parth time Vs. METIS " << parth_time << " - " << metis_time
            << " Reuse: " << parth.getReuse() << std::endl;
}

void ParthTestUtils::testAddingDOFs(std::string hessian_name,
                                    std::string mesh_name) {
  //************** Load the Mesh  *************
  Eigen::SparseMatrix<double> mesh_csc;
  if (!Eigen::loadMarket(mesh_csc, mesh_name)) {
    std::cerr << "File " << mesh_name << " is not found" << std::endl;
  }

  Eigen::SparseMatrix<double> lower_A_csc;
  if (!Eigen::loadMarket(lower_A_csc, hessian_name)) {
    std::cerr << "File " << hessian_name << " is not found" << std::endl;
  }

  //************** setup Parth *************
  PARTH::Parth parth;
  parth.setReorderingType(PARTH::ReorderingType::METIS);
  parth.setVerbose(true);
  parth.setNDLevels(2);
  parth.setNumberOfCores(4);
  //************ Check the adding and deleting edge functionality ***********
  // Reuse of 1 when separator and left and right sub-meshes are connected
  std::vector<int> parth_perm(lower_A_csc.rows());
  parth.setMeshPointers(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
                        mesh_csc.innerIndexPtr());
  parth.computePermutation(parth_perm);
  assert(parth.hmd.num_HMD_nodes == 7);

  for (int i = 0; i < parth.hmd.num_HMD_nodes; i++) {
    std::cout << "HMD node: " << i << " DOFs size: "
              << (parth.hmd.HMD_tree[i].DOFs.size() * 1.0) / mesh_csc.rows()
              << " Edges: "
              << parth.hmd.getTotalEdgesInHMDNode(i, parth.M_n, parth.Mp) *
                     1.0 / mesh_csc.nonZeros()
              << std::endl;
  }

  // ---------------------------------- Adding DOFs
  int offset = mesh_csc.rows();
  std::vector<int> dof_mapper;
  std::vector<int> offset_to_add = {400};
  std::vector<int> Mp_new;
  std::vector<int> Mi_new;
  this->AddDOF(mesh_csc.rows(), mesh_csc.outerIndexPtr(),
               mesh_csc.innerIndexPtr(), 1, 100, Mp_new, Mi_new, offset_to_add,
               dof_mapper);
  std::vector<std::pair<int, int>> edge_set;
  // Compute old to new mapper
  std::vector<int> old_to_new_dof_mapper(mesh_csc.rows());
  for (int i = 0; i < dof_mapper.size(); i++) {
    if (dof_mapper[i] != -1) {
      old_to_new_dof_mapper[dof_mapper[i]] = i;
    }
  }
  for (int i = 0; i < dof_mapper.size(); i++) {
    if (dof_mapper[i] == -1) {
      int hmd_node = parth.hmd.DOF_to_HMD_node[5000];
      std::cout << "The node is: " << hmd_node << std::endl;
      int second_1 = parth.hmd.HMD_tree[hmd_node].DOFs[0];
      int second_2 = parth.hmd.HMD_tree[hmd_node].DOFs[1];

      edge_set.emplace_back(i, old_to_new_dof_mapper[second_1]);
      edge_set.emplace_back(i, old_to_new_dof_mapper[second_2]);
    }
  }

  std::vector<int> Mp_new_2;
  std::vector<int> Mi_new_2;
  std::vector<double> Mx_new_2;

  this->addEdgeToMesh(Mp_new.size() - 1, Mp_new.data(), Mi_new.data(), Mp_new_2,
                      Mi_new_2, edge_set);
  parth.setMeshPointers(Mp_new_2.size() - 1, Mp_new_2.data(), Mi_new_2.data());
  parth.setNewToOldDOFMap(dof_mapper);

  std::cout << "--------------------- Adding DOFs " << std::endl;
  // Measure the time
  Mx_new_2.resize(Mi_new.size());
  // Add random values for Mx_test
  for (int i = 0; i < Mi_new.size(); i++) {
    Mx_new_2[i] = rand() % 100;
  }
  double parth_time = omp_get_wtime();
  parth.computePermutation(parth_perm);
  assert(parth.mesh_perm.size() == Mp_new_2.size() - 1);
  int parth_nnz =
      this->getLNNZ(Mi_new_2.size(), Mp_new_2.size() - 1, Mp_new_2.data(),
                    Mi_new_2.data(), Mx_new_2.data(), parth.mesh_perm.data());
  parth_time = omp_get_wtime() - parth_time;

  for (int i = 0; i < parth.hmd.num_HMD_nodes; i++) {
    std::cout << "HMD node: " << i << " DOFs size: "
              << (parth.hmd.HMD_tree[i].DOFs.size()) * 1.0 /
                     (Mp_new_2.size() - 1)
              << " Edges: "
              << parth.hmd.getTotalEdgesInHMDNode(i, parth.M_n, parth.Mp) *
                     1.0 / Mi_new_2.size()
              << std::endl;
  }
  // Measure the time for METIS
  double metis_time = omp_get_wtime();
  int metis_nnz =
      this->getLNNZ_METIS(Mi_new_2.size(), Mp_new_2.size() - 1, Mp_new_2.data(),
                          Mi_new_2.data(), Mx_new_2.data());
  metis_time = omp_get_wtime() - metis_time;
  // Print the difference in quality
  std::cout
      << "Difference between METIS and PARTH -> the more is better for parth: "
      << (double)(metis_nnz - parth_nnz) / (double)metis_nnz << std::endl;
  // Print the time
  std::cout << "The Parth time Vs. METIS " << parth_time << " - " << metis_time
            << " Reuse: " << parth.getReuse() << std::endl;

  // Save Parth permutation
  std::ofstream file;
  file.open("/home/behrooz/Desktop/IPC_Project/SolverTestBench/Analysis/"
            "PermutationQualityVisualization/data/parth_perm.txt");
  if (file.is_open()) {
    file << parth.mesh_perm.size()
         << "\n"; // write the vector size on the first line.
    for (auto &n : parth.mesh_perm) { // iterate over the vector and write each
                                      // number on a new line.
      file << n << "\n";
    }
    file.close();
  } else {
    std::cout << "Unable to open file";
  }

  // Save matrix
  Eigen::SparseMatrix<double> parth_matrix(Mp_new_2.size() - 1,
                                           Mp_new_2.size() - 1);
  parth_matrix.reserve(Mi_new_2.size());
  for (int i = 0; i < Mp_new_2.size() - 1; i++) {
    for (int j = Mp_new_2[i]; j < Mp_new_2[i + 1]; j++) {
      parth_matrix.insert(i, Mi_new_2[j]) = Mx_new_2[j];
    }
  }
  assert(parth_matrix.rows() == parth.mesh_perm.size());
  Eigen::saveMarket(
      parth_matrix,
      "/home/behrooz/Desktop/IPC_Project/SolverTestBench/Analysis/"
      "PermutationQualityVisualization/data/mesh.mtx");
  parth.printTiming();
}

void ParthTestUtils::testAddingDeletingDOFs(std::string hessian_name,
                                            std::string mesh_name) {

  this->testDeletingDOFs(hessian_name, mesh_name);

  this->testAddingDOFs(hessian_name, mesh_name);

  // TODO: Adding and deleting DOFs
}

void ParthTestUtils::computeTheAddedRemovedEdgesForDOF(
    int dof_id, int *Mi_new, int new_start, int new_end, int *Mi_old,
    int old_start, int old_end, std::vector<std::pair<int, int>> &edge_set) {
  // create pairs of all edges for new mesh
  std::set<std::pair<int, int>> newEdges;
  for (int i = new_start; i < new_end; i++) {
    int nbr = Mi_new[i];
    if (nbr == dof_id) {
      continue;
    }
    if (dof_id > nbr) {
      newEdges.insert(std::pair<int, int>(dof_id, nbr));
    }
  }
  // create pairs of all edges for old mesh
  std::set<std::pair<int, int>> oldEdges;
  for (int i = old_start; i < old_end; i++) {
    int nbr = Mi_old[i];
    if (nbr == dof_id) {
      continue;
    }
    if (dof_id > nbr) {
      oldEdges.insert(std::pair<int, int>(dof_id, nbr));
    }
  }
  // find the difference between the two sets
  std::set_difference(newEdges.begin(), newEdges.end(), oldEdges.begin(),
                      oldEdges.end(),
                      std::inserter(edge_set, edge_set.begin()));

  std::set_difference(oldEdges.begin(), oldEdges.end(), newEdges.begin(),
                      newEdges.end(),
                      std::inserter(edge_set, edge_set.begin()));
}

void ParthTestUtils::addEdgeToMesh(int M_n, int *Mp, int *Mi,
                                   std::vector<int> &Mp_new,
                                   std::vector<int> &Mi_new,
                                   std::vector<std::pair<int, int>> &edge_set) {
  // Converting edge_set into a 2D array
  std::vector<std::vector<int>> edge_array(M_n);
  for (auto &edge : edge_set) {
    edge_array[edge.first].push_back(edge.second);
    edge_array[edge.second].push_back(edge.first);
  }

  Mp_new.clear();
  Mi_new.clear();
  // Adding the edges to the mesh
  int nnz = 0;
  Mp_new.push_back(0);
  for (int i = 0; i < M_n; i++) {
    // Adding the previous edges
    for (int j = Mp[i]; j < Mp[i + 1]; j++) {
      Mi_new.push_back(Mi[j]);
      nnz++;
    }
    // Adding the new edges
    for (auto &edge : edge_array[i]) {
      Mi_new.push_back(edge);
      nnz++;
    }
    int start = Mp_new.back();
    Mp_new.push_back(nnz);
    // Sorting the edges indices of each col in CSC format
    std::sort(Mi_new.begin() + start, Mi_new.begin() + Mp_new.back());
  }

  assert(Mp_new.size() == M_n + 1);
  assert(Mp_new.back() == Mi_new.size());
#ifndef NDEBUG
  // Check whether the edges are added and only those edges are added
  for (int i = 0; i < M_n; i++) {
    // Compute the difference between edges of the new and old mesh
    std::vector<std::pair<int, int>> changed_edges;
    this->computeTheAddedRemovedEdgesForDOF(i, Mi_new.data(), Mp_new[i],
                                            Mp_new[i + 1], Mi, Mp[i], Mp[i + 1],
                                            changed_edges);
    for (auto &edge : changed_edges) {
      int first = edge.first;
      int second = edge.second;
      // Check whether the edge is added
      assert(std::find(edge_array[first].begin(), edge_array[first].end(),
                       second) != edge_array[first].end());
      assert(std::find(edge_array[second].begin(), edge_array[second].end(),
                       first) != edge_array[second].end());
    }
  }
#endif
}

void ParthTestUtils::createEdges(
    PARTH::Parth &parth, ///<[in] Parth object
    std::vector<std::pair<int, int>>
        &HMD_edge_set,      ///<[in] Edge set in HMD nodes
    int sample_range_start, ///<[in] minimum number of DOFs edges per each
                            ///< HMD_edge_set
    int sample_range_end,   ///<[in] maximum number of DOFs edges per each
                            ///< HMD_edge_set
    std::vector<std::pair<int, int>> &edge_set ///<[out] Edge set across DOFs
) {
  // Creating the edges
  edge_set.clear();

  assert(sample_range_end > sample_range_start);
  assert(sample_range_start > 0);

  for (auto &edge : HMD_edge_set) {
    int HMD_node_1 = std::get<0>(edge);
    int HMD_node_2 = std::get<1>(edge);
    assert(HMD_node_1 < parth.hmd.HMD_tree.size());
    assert(HMD_node_2 < parth.hmd.HMD_tree.size());

    int num_edges =
        rand() % (sample_range_end - sample_range_start) + sample_range_start;
    for (int i = 0; i < num_edges; i++) {
      // Get a random dof from the HMD_node_1 assigned DOFs
      std::vector<int> &HMD_node_1_DOFs = parth.hmd.HMD_tree[HMD_node_1].DOFs;
      int DOFs_1_size = HMD_node_1_DOFs.size();
      int DOF_1 = HMD_node_1_DOFs[rand() % DOFs_1_size];
      // Get a random dof from the HMD_node_2 assigned DOFs
      std::vector<int> &HMD_node_2_DOFs = parth.hmd.HMD_tree[HMD_node_2].DOFs;
      int DOFs_2_size = HMD_node_2_DOFs.size();
      if (DOFs_2_size == 0) {
        continue;
      }
      int DOF_2 = HMD_node_2_DOFs[rand() % DOFs_2_size];
      // Make sure that an edge is not created between the same DOF
      while (DOF_1 == DOF_2) {
        DOF_2 = HMD_node_2_DOFs[rand() % DOFs_2_size];
      }
      // Add the edge DOF_1->DOF_2 and DOF_2->DOF_1 to maintain symmetric mesh
      edge_set.push_back({DOF_1, DOF_2});
    }
  }
}

void ParthTestUtils::deleteDOF(PARTH::Parth &parth, int M_n, int *Mp, int *Mi,
                               int sample_range_start, int sample_range_end,
                               std::vector<int> &Mp_new,
                               std::vector<int> &Mi_new,
                               std::vector<int> &HMD_nodes_to_delete_dof,
                               std::vector<int> &dof_mapper) {
  // Computing the DOF_set to delete
  std::set<int> DOF_set;
  for (auto &HMD_node_idx : HMD_nodes_to_delete_dof) {
    int num_DOFs =
        rand() % (sample_range_end - sample_range_start) + sample_range_start;
    for (int i = 0; i < num_DOFs; i++) {
      // Get a random dof from the HMD_node_1 assigned DOFs
      std::vector<int> &HMD_node_1_DOFs = parth.hmd.HMD_tree[HMD_node_idx].DOFs;
      int DOFs_1_size = HMD_node_1_DOFs.size();
      int DOF_1 = HMD_node_1_DOFs[rand() % DOFs_1_size];
      DOF_set.insert(DOF_1);
    }
  }

  // Deleting the DOF from the mesh
  std::vector<int> update_value(M_n, 0);
  for (auto &dof : DOF_set) {
    for (int l = dof; l < M_n; l++) {
      update_value[l]--;
    }
  }

  dof_mapper.clear();
  Mi_new.clear();
  Mp_new.clear();
  int nnz = 0;
  Mp_new.push_back(nnz);
  for (int i = 0; i < M_n; i++) {
    if (std::find(DOF_set.begin(), DOF_set.end(), i) != DOF_set.end()) {
      continue;
    }
    // Adding the previous edges
    for (int j = Mp[i]; j < Mp[i + 1]; j++) {
      if (std::find(DOF_set.begin(), DOF_set.end(), Mi[j]) == DOF_set.end()) {
        Mi_new.push_back(Mi[j]);
        nnz++;
      }
    }
    Mp_new.push_back(nnz);
    dof_mapper.emplace_back(i);
  }

#ifndef NDEBUG
  assert(Mi_new.size() == nnz);
  assert(Mp_new.back() == nnz);
  int total_deleted_edges = 0;
  for (int i = 0; i < M_n; i++) {
    if (std::find(DOF_set.begin(), DOF_set.end(), i) != DOF_set.end()) {
      total_deleted_edges += (Mp[i + 1] - Mp[i]);
    }
  }
  assert(nnz + total_deleted_edges * 2 - Mp[M_n] == 0);
#endif

  std::vector<int> old_to_new_dof_mapper(M_n, -1);
  for (int n = 0; n < dof_mapper.size(); n++) {
    // If the mapping exist, the node exist
    assert(dof_mapper[n] < M_n); // dof ids are zero based
    assert(dof_mapper[n] != -1);
    old_to_new_dof_mapper[dof_mapper[n]] = n;
  }

#ifndef NDEBUG
  for (auto &dof : DOF_set) {
    assert(old_to_new_dof_mapper[dof] == -1);
  }
#endif

  // Mapping Mi_new with updated labels
  for (int i = 0; i < Mi_new.size(); i++) {
    assert(old_to_new_dof_mapper[Mi_new[i]] != -1);
    Mi_new[i] = old_to_new_dof_mapper[Mi_new[i]];
    assert(Mi_new[i] < Mp_new.size() - 1);
  }

#ifndef NDEBUG
  for (int i = 0; i < dof_mapper.size(); i++) {
    if (dof_mapper[i] != -1) {
      assert(dof_mapper[i] + update_value[dof_mapper[i]] == i);
    }
  }
#endif
  //  // Print deleted dofs
  //  std::cout << "Deleted DOFs are: ";
  //  for (auto &dof : DOF_set) {
  //    std::cout << dof << " ";
  //  }
  //  std::cout << std::endl;
}

void ParthTestUtils::AddDOF(int M_n, int *Mp, int *Mi, int sample_range_start,
                            int sample_range_end, std::vector<int> &Mp_new,
                            std::vector<int> &Mi_new,
                            std::vector<int> offset_to_add,
                            std::vector<int> &dof_mapper) {
  Mp_new.clear();
  std::vector<bool> new_flag;
  std::sort(offset_to_add.begin(), offset_to_add.end());
  int offset_cnt = 0;
  int total_added_nodes = 0;
  for (int i = 0; i <= M_n; i++) {
    int first_offset = offset_to_add[offset_cnt];
    if (first_offset == i) {
      int num_nodes =
          rand() % (sample_range_end - sample_range_start) + sample_range_start;
      int Mp_value = Mp[i];
      for (int j = 0; j < num_nodes; j++) {
        Mp_new.push_back(Mp_value);
        new_flag.push_back(true);
        total_added_nodes++;
      }
      Mp_new.push_back(Mp_value); // The existed node
      if (first_offset != M_n) {
        new_flag.push_back(false);
      }
      offset_cnt++;
      if (offset_cnt == offset_to_add.size()) {
        offset_cnt--;
      }
    } else {
      Mp_new.push_back(Mp[i]);
      if (i != M_n) {
        new_flag.push_back(false);
      }
    }
  }
  // Create the mapper
  dof_mapper.clear();
  assert(Mp_new.size() - 1 == M_n + total_added_nodes);
  int cnt = 0;
  for (int i = 0; i < new_flag.size(); i++) {
    if (new_flag[i]) {
      dof_mapper.push_back(-1);
    } else {
      dof_mapper.push_back(cnt++);
    }
  }
  assert(Mp_new.size() == new_flag.size() + 1);
  Mi_new.clear();
  // Compute old to new mapping
  std::vector<int> old_to_new_dof_mapper(M_n, -1);
  for (int n = 0; n < dof_mapper.size(); n++) {
    // If the mapping exist, the node exist
    assert(dof_mapper[n] < M_n); // dof ids are zero based
    if (dof_mapper[n] != -1) {
      old_to_new_dof_mapper[dof_mapper[n]] = n;
    }
  }
  for (int i = 0; i < Mp[M_n]; i++) {
    Mi_new.push_back(old_to_new_dof_mapper[Mi[i]]);
  }
  assert(dof_mapper.size() == Mp_new.size() - 1);
}

} // namespace PARTH

#include "ParthTestUtils.h"
