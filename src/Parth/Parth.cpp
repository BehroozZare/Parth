//
// Created by behrooz zare on 2024-04-07.
//

#include "Parth.h"
#include "omp.h"
#include <iostream>

namespace PARTH {

//======================== Options Functions ========================
void Parth::setReorderingType(PARTH::ReorderingType type) {
  this->reorder_type = type;
  hmd.reorder_type = reorder_type;
}

PARTH::ReorderingType Parth::getReorderingType() const { return reorder_type; }

void Parth::setVerbose(bool verbose) { this->verbose = verbose; }

bool Parth::getVerbose() const { return verbose; }

void Parth::setNDLevels(int num) {
  this->ND_levels = num;
  hmd.num_levels = ND_levels;
}

int Parth::getNDLevels() const { return this->ND_levels; }

///--------------------------------------------------------------------------
/// setNumberOfCores - Set the number of cores for region parallelism
///--------------------------------------------------------------------------
void Parth::setNumberOfCores(int num_cores) { this->num_cores = num_cores; }

int Parth::getNumberOfCores() const { return this->num_cores; }

///--------------------------------------------------------------------------
/// setNumSubMesh - Set the number of nested dissection levels for HMD
///--------------------------------------------------------------------------
void Parth::setRelocatedLevel(int lvl) { integrator.relocate_lvl = lvl; }
int Parth::getRelocatedLevel() const { return integrator.relocate_lvl; }

void Parth::setLagLevel(int lvl) { integrator.lag_lvl = lvl; }
int Parth::getLagLevel() const { return integrator.lag_lvl; }

Parth::Parth() {
  // SET DEFAULT OPTIONS
  setReorderingType(ReorderingType::METIS);
  setVerbose(true);
  setNDLevels(8);
  setNumberOfCores(10);

  Mp = nullptr;
  Mi = nullptr;
  M_n = 0;
  M_n_prev = 0;
}

Parth::~Parth() {}

void Parth::clearParth() {
  Mp = nullptr;
  Mi = nullptr;
  M_n = 0;

  M_n_prev = 0;

  mesh_perm.clear();
  changed_dof_edges.clear();

  hmd.clearHMD();
  integrator.clearIntegrator();
}

void Parth::setMesh(int n, int *Mp, int *Mi, std::vector<int> &map) {
  assert(Mp != nullptr);
  assert(Mi != nullptr);
  assert(n != 0);
  this->Mp = Mp;
  this->Mi = Mi;
  this->M_n = n;
  if (map.empty()) {
    new_to_old_map.clear();
  } else {
    this->setNewToOldDOFMap(map);
  }
}

void Parth::setMesh(int n, int *Mp, int *Mi) {
  assert(Mp != nullptr);
  assert(Mi != nullptr);
  assert(n != 0);
  this->Mp = Mp;
  this->Mi = Mi;
  this->M_n = n;
  this->new_to_old_map.clear();
}

void Parth::setMatrix(int N, int *Ap, int* Ai, int dim) {
  assert(Ap != nullptr);
  assert(Ai != nullptr);
  assert(N != 0);
  this->sim_dim = dim;

  //Convert the matrix into mesh and init the pointers
  this->computeMeshFromMatrix(Ap, Ai, N, dim);
}

void Parth::setMatrix(int N, int *Ap, int* Ai, std::vector<int> &map, int dim) {
  assert(Ap != nullptr);
  assert(Ai != nullptr);
  assert(N != 0);
  this->sim_dim = dim;

  //Convert the matrix into mesh and init the pointers
  this->computeMeshFromMatrix(Ap, Ai, N, dim);

  if (map.empty()) {
    new_to_old_map.clear();
  } else {
    this->setNewToOldDOFMap(map);
  }
}

void Parth::setNewToOldDOFMap(std::vector<int> &map) {
  this->new_to_old_map = map;
  this->computeOldToNewDOFMap();
}

void Parth::computeOldToNewDOFMap() {
  added_dofs.clear();
  old_to_new_map.clear();
  removed_dofs.clear();
  if (M_n_prev == 0) {
    new_to_old_map.clear();
    return;
  }
  old_to_new_map.resize(M_n_prev, -1);
  for (int n = 0; n < M_n; n++) {
    if (this->new_to_old_map[n] != -1) {
      assert(this->new_to_old_map[n] < M_n_prev);
      this->old_to_new_map[this->new_to_old_map[n]] = n;
    } else {
      added_dofs.emplace_back(n);
    }
  }

  for (int j = 0; j < M_n_prev; j++) {
    if (this->old_to_new_map[j] == -1) {
      removed_dofs.emplace_back(j);
    }
  }

#ifndef NDEBUG
  std::vector<bool> visited(M_n, false);
  for (int i = 0; i < old_to_new_map.size(); i++) {
    if (old_to_new_map[i] != -1) {
      assert(old_to_new_map[i] < M_n);
      assert(!visited[old_to_new_map[i]]);
      visited[old_to_new_map[i]] = true;
    }
  }

  if (added_dofs.empty()) {
    for (int i = 0; i < visited.size(); i++) {
      assert(visited[i]);
    }
  }

  assert(M_n == M_n_prev + added_dofs.size() - removed_dofs.size());

#endif

  if (verbose) {
    std::cout << "+++ PARTH: number of DOF added nodes are +++"
              << added_dofs.size() << " From " << M_n << std::endl;
    std::cout << "+++ PARTH: number of DOF removed nodes are +++"
              << removed_dofs.size() << " From " << M_n << std::endl;
  }
  if (added_dofs.size() >
      0.4 * M_n) { // TODO: This can be some user defined threshold
    removed_dofs.clear();
    added_dofs.clear();
    old_to_new_map.clear();
    new_to_old_map.clear();
    hmd.HMD_init = false;
  }
}

void Parth::computePermutation(std::vector<int> &perm, int dim) {
  if (verbose) {
    std::cout << "+++ PARTH: Permutation Computation +++" << std::endl;
  }

  // Resting the timing
  init_time = 0.0;
  map_mesh_to_matrix_computation_time = 0.0;
  dof_change_integrator_time = 0.0;
  change_computation_time = 0.0;
  compute_permutation_time = 0.0;

  // Reset the statistics
  this->integrator.reuse_ratio = 0;
  this->changed_dof_edges.clear();

  sim_dim = dim;

  hmd.verbose = getVerbose();
  if (M_n != M_n_prev && new_to_old_map.empty()) {
    hmd.HMD_init = false;
  }

  perm.resize(M_n * dim);

  // Initialize the HMD tree
  if (!hmd.HMD_init) {
    init_time = omp_get_wtime();
    hmd.initHMD(getNDLevels(), M_n, Mp, Mi, mesh_perm);
    init_time = omp_get_wtime() - init_time;
    map_mesh_to_matrix_computation_time = omp_get_wtime();
    mapMeshPermToMatrixPerm(mesh_perm, perm, dim);
    map_mesh_to_matrix_computation_time =
        omp_get_wtime() - map_mesh_to_matrix_computation_time;
    integrator.reuse_ratio = 0;
  } else {
    init_time = 0.0;
    map_mesh_to_matrix_computation_time = 0.0;
    // Activate Immobilizer
    //  Compute Local changes points
    if (!new_to_old_map.empty()) {
      assert(new_to_old_map.size() == M_n);
      assert(old_to_new_map.size() == M_n_prev);
      dof_change_integrator_time = omp_get_wtime();
      integrator.integrateDOFChanges(hmd, new_to_old_map, old_to_new_map,
                                     added_dofs, removed_dofs, M_n, Mp, Mi,
                                     M_n_prev, mesh_perm);
      dof_change_integrator_time = omp_get_wtime() - dof_change_integrator_time;

      change_computation_time = omp_get_wtime();

      integrator.computeChangedEdges(hmd, M_n, Mp, Mi, M_n_prev, Mp_prev.data(),
                                     Mi_prev.data(), new_to_old_map.data(),
                                     old_to_new_map.data(), changed_dof_edges);
      change_computation_time = omp_get_wtime() - change_computation_time;
    } else {
      dof_change_integrator_time = 0.0;
      change_computation_time = omp_get_wtime();
      integrator.computeChangedEdges(hmd, M_n, Mp, Mi, M_n_prev, Mp_prev.data(),
                                     Mi_prev.data(), nullptr, nullptr,
                                     changed_dof_edges);
      change_computation_time = omp_get_wtime() - change_computation_time;
    }
    std::cout << "+++ PARTH: number of bad edges are "
              << changed_dof_edges.size() << std::endl;
    assert(mesh_perm.size() == M_n);
    compute_permutation_time = omp_get_wtime();
    if (changed_dof_edges.empty()) {
      mapMeshPermToMatrixPerm(mesh_perm, perm, dim);
      integrator.reuse_ratio = 1;
    } else {
      // call Integrator routine
      integrator.getGlobalPerm(hmd, M_n, Mp, Mi, changed_dof_edges,
                               activate_aggressive_reuse, lagging, mesh_perm);
      mapMeshPermToMatrixPerm(mesh_perm, perm, dim);
    }
    compute_permutation_time = omp_get_wtime() - compute_permutation_time;
  }

  M_n_prev = M_n;
  Mp_prev.resize(M_n_prev + 1);
  Mi_prev.resize(Mp[M_n]);
  std::copy(Mp, Mp + M_n + 1, Mp_prev.data());
  std::copy(Mi, Mi + Mp[M_n], Mi_prev.data());
  new_to_old_map.clear();
  old_to_new_map.clear();
  added_dofs.clear();
  removed_dofs.clear();
}

///--------------------------------------------------------------------------
/// mapMeshPermToMatrixPerm - Map the mesh permutation to matrix permutation ->
/// Final stage of assembler
///--------------------------------------------------------------------------
void Parth::mapMeshPermToMatrixPerm(std::vector<int> &mesh_perm,
                                    std::vector<int> &matrix_perm, int dim) {
#ifndef NDEBUG // TODO: DELETE this
  std::vector<bool> mesh_perm_flag(M_n, false);
  for (int i = 0; i < M_n; i++) {
    mesh_perm_flag[mesh_perm[i]] = true;
  }
  double invalid_perm = 0;
  for (int i = 0; i < M_n; i++) {
    if (!mesh_perm_flag[i]) {
      invalid_perm++;
    }
  }

  if (invalid_perm != 0) {
    std::cerr << "Your permutation doesn't cover: " << invalid_perm / (M_n)
              << std::endl;
    assert(invalid_perm == 0);
  }
#endif
  if (dim == -1) {
    dim = sim_dim;
  }

  if (dim == 1) {
    matrix_perm = mesh_perm;
  }

#pragma omp parallel for // TODO: only fix the region with change
  for (int idx = 0; idx < M_n; idx++) {
    for (int d = 0; d < dim; d++) {
      matrix_perm[idx * dim + d] = mesh_perm[idx] * dim + d;
    }
  }

#ifndef NDEBUG // TODO: DELETE this
  std::vector<bool> matrix_perm_flag(M_n * dim, false);
  for (int i = 0; i < M_n * dim; i++) {
    matrix_perm_flag[matrix_perm[i]] = true;
  }
  invalid_perm = 0;
  for (int i = 0; i < M_n * dim; i++) {
    if (!matrix_perm_flag[i]) {
      invalid_perm++;
    }
  }
  if (invalid_perm != 0) {
    std::cerr << "Your permutation doesn't cover: " << invalid_perm / (M_n * 3)
              << std::endl;
    assert(invalid_perm == 0);
  }
#endif
}

double Parth::getReuse() { return integrator.reuse_ratio; }

int Parth::getNumChanges() { return changed_dof_edges.size(); }

void Parth::resetTimers() {
  init_time = 0.0;
  map_mesh_to_matrix_computation_time = 0.0;
  dof_change_integrator_time = 0.0;
  change_computation_time = 0.0;
  compute_permutation_time = 0.0;
}
void Parth::printTiming() {
  std::cout << "+++ PARTH: Timing START +++" << std::endl;
  std::cout << "+++ PARTH: Init Time: " << init_time << " sec" << std::endl;
  std::cout << "+++ PARTH: DOF Change Integration Time: "
            << dof_change_integrator_time << " sec" << std::endl;
  std::cout << "+++ PARTH: Change Computation Time: " << change_computation_time
            << " sec" << std::endl;
  std::cout << "+++ PARTH: Compute Permutation Time: "
            << compute_permutation_time << " sec" << std::endl;
  std::cout << "+++ PARTH: Map Mesh to Matrix Computation Time: "
            << map_mesh_to_matrix_computation_time << " sec" << std::endl;
  std::cout << "+++ PARTH: Overall reuse: " << getReuse() << std::endl;
  std::cout << "+++ PARTH: Timing END +++" << std::endl;
}


void Parth::computeMeshFromMatrix(int* Ap, int* Ai, int N, int dim) {  
  assert(N % dim == 0);
  this->Mp_vec.clear();
  this->Mp_vec.resize(N / dim + 1, 0);
  std::vector<std::tuple<int, int>> coefficients;
  for (int c = 0; c < N; c += dim) {
    assert((Ap[c + 1] - Ap[c]) % dim == 0);
    for (int r_ptr = Ap[c]; r_ptr < Ap[c + 1]; r_ptr += dim) {
      int r = Ai[r_ptr];
      int mesh_c = c / dim;
      int mesh_r = r / dim;
      if (mesh_c != mesh_r) {
        coefficients.emplace_back(mesh_c, mesh_r);
        coefficients.emplace_back(mesh_r, mesh_c);
      }
    }
  }

  //Remove duplicates
  std::sort(coefficients.begin(), coefficients.end());
  coefficients.erase(std::unique(coefficients.begin(), coefficients.end()), coefficients.end());

  for (int i = 0; i < coefficients.size(); i++) {
    this->Mp_vec[std::get<0>(coefficients[i]) + 1]++;
  }
  for (int i = 1; i < this->Mp_vec.size(); i++) {
    this->Mp_vec[i] += this->Mp_vec[i - 1];
  }

  this->Mi_vec.resize(this->Mp_vec.back());
  std::vector<int> Mp_vec_cnt(this->Mp_vec.size(), 0);
  for (int i = 0; i < coefficients.size(); i++) {
    int row = std::get<0>(coefficients[i]);
    int col = std::get<1>(coefficients[i]);
    this->Mi_vec[this->Mp_vec[row] + Mp_vec_cnt[row]] = col;
    Mp_vec_cnt[row]++;
  }

  #ifndef NDEBUG
  //Make sure that each row is sorted
  for(int r = 0; r < this->Mp_vec.size(); r++) {
    for (int i = this->Mp_vec[r]; i < this->Mp_vec[r + 1] - 1; i++) {
      assert(this->Mi_vec[i] < this->Mi_vec[i + 1]);
    }
  }
  #endif

  int M_n = this->Mp_vec.size() - 1;
  this->Mp = this->Mp_vec.data();
  this->Mi = this->Mi_vec.data();
  this->M_n = M_n;
  return;
}


} // namespace PARTH
