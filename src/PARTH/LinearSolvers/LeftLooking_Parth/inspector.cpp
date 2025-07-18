//
// Created by behrooz on 28/09/22.
//

#include "Barb.h"
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
namespace PARTH {

void ParthSolver::hessianAnalyze() {
  if (Options().getVerbose()) {
    std::cout << "+++ PARTH: Cholmod default Analysis +++" << std::endl;
  }
  chol_L = cholmod_analyze(chol_A, &A_cm);
}

void ParthSolver::meshAnalyze() {
  if (Options().getVerbose()) {
    std::cout << "+++ PARTH: Mesh Analysis +++" << std::endl;
  }
  if (!this->mesh_init) {
    std::cerr << "+++ PARTH: Please initialize the mesh first" << std::endl;
    return;
  }

  std::vector<int> contact_points;
  computeContacts(contact_points);
  if (!contact_points.empty()) {
    mesh_perm.resize(M_n);
    std::vector<int> mesh_perm_inv(M_n);
    regions.metisNodeNDPermutation(M_n, Mp, Mi, mesh_perm.data(),
                                   mesh_perm_inv.data());
    // Assemble the permutation
    for (int j = 0; j < M_n; j++) {
      hessian_perm[j * 3] = mesh_perm[j] * 3;
      hessian_perm[j * 3 + 1] = mesh_perm[j] * 3 + 1;
      hessian_perm[j * 3 + 2] = mesh_perm[j] * 3 + 2;
    }
  }

#ifndef NDEBUG // TODO: DELETE this
  std::vector<bool> perm_flag(A_n, false);
  for (int i = 0; i < A_n; i++) {
    perm_flag[hessian_perm[i]] = true;
  }
  double invalid_perm = 0;
  for (int i = 0; i < A_n; i++) {
    if (!perm_flag[i]) {
      invalid_perm++;
      assert(true);
    }
  }
  if (invalid_perm != 0) {
    std::cerr << "Your permutation doesn't cover: " << invalid_perm / A_n
              << std::endl;
  }
#endif

  //  Apply the reordering and perform the analysis
  A_cm.nmethods = 1;
  A_cm.method[0].ordering = CHOLMOD_GIVEN;
  chol_L =
      cholmod_analyze_p_custom(chol_A, hessian_perm.data(), NULL, 0, &A_cm);

  if (Options().getVerbose()) {
    int mesh_cnt = 0;
    std::cout << "+++ PARTH: The mesh for region " << mesh_cnt << std::endl;
    if (contact_points.empty()) {
      std::cout << "+++ PARTH: Reuse Status: REUSED" << std::endl;
    } else {
      std::cout << "+++ PARTH: Reuse Status: NOT REUSED" << std::endl;
    }
    mesh_cnt++;
    std::cout << "++++++++++++++++" << std::endl;
  }
}

void ParthSolver::getPermutationWithReuse(std::vector<int> &perm) {
  std::cout << "+++ PARTH: Permutation Computation +++" << std::endl;
  if (chol_L != nullptr) {
    cholmod_free_factor(&chol_L, &A_cm);
  }

  regions.verbose = opt.getVerbose();
  if (M_n != M_n_prev) {
    regions.regions_init = false;
  }

  perm.resize(M_n * 3);
  if (!regions.regions_init) {
    regions.initRegions(opt.getNumRegions(), M_n, Mp, Mi, mesh_perm);
#pragma omp parallel for // TODO: only fix the region with change
    for (int idx = 0; idx < M_n; idx++) {
      perm[idx * 3] = mesh_perm[idx] * 3;
      perm[idx * 3 + 1] = mesh_perm[idx] * 3 + 1;
      perm[idx * 3 + 2] = mesh_perm[idx] * 3 + 2;
    }
  } else {
    std::vector<int> contact_points;
    // Compute Local changes points
    computeContacts(contact_points);

    regions.assignMeshPtr(M_n, Mp, Mi);
    regions.getGlobalPerm(contact_points, mesh_perm);
#pragma omp parallel for // TODO: only fix the region with change
    for (int idx = 0; idx < M_n; idx++) {
      perm[idx * 3] = mesh_perm[idx] * 3;
      perm[idx * 3 + 1] = mesh_perm[idx] * 3 + 1;
      perm[idx * 3 + 2] = mesh_perm[idx] * 3 + 2;
    }
  }

  M_n_prev = M_n;
  M_nnz_prev = Mp[M_n];
  Mp_prev.resize(M_n_prev + 1);
  Mi_prev.resize(M_nnz_prev);
  std::copy(Mp, Mp + M_n + 1, Mp_prev.data());
  std::copy(Mi, Mi + M_nnz_prev, Mi_prev.data());

#ifndef NDEBUG // TODO: DELETE this
  std::vector<bool> perm_flag(A_n, false);
  for (int i = 0; i < A_n; i++) {
    perm_flag[perm[i]] = true;
  }
  double invalid_perm = 0;
  for (int i = 0; i < A_n; i++) {
    if (!perm_flag[i]) {
      invalid_perm++;
      assert(false);
    }
  }
  if (invalid_perm != 0) {
    std::cerr << "Your permutation doesn't cover: " << invalid_perm / A_n
              << std::endl;
    assert(invalid_perm == 0);
  }
#endif
}

void ParthSolver::SimulationAwareAnalyze() {
  if (Options().getVerbose()) {
    std::cout << "+++ PARTH: Simulation Analysis +++" << std::endl;
  }
  if (!this->mesh_init) {
    std::cerr << "Please initialize the mesh first" << std::endl;
    return;
  }
  regions.verbose = opt.getVerbose();
  if (M_n != M_n_prev) {
    regions.regions_init = false;
  }

  if (!regions.regions_init) {
    regions.initRegions(opt.getNumRegions(), M_n, Mp, Mi, mesh_perm);
#pragma omp parallel for // TODO: only fix the region with change
    for (int idx = 0; idx < M_n; idx++) {
      hessian_perm[idx * 3] = mesh_perm[idx] * 3;
      hessian_perm[idx * 3 + 1] = mesh_perm[idx] * 3 + 1;
      hessian_perm[idx * 3 + 2] = mesh_perm[idx] * 3 + 2;
    }
  } else {
    std::vector<int> contact_points;
    // Compute Local changes points
    computeContacts(contact_points);
    // If number of contact points are zero and we have previous factor
    // use the symbolic information within that factor
    if (Options().getNumericReuseType() ==
            NumericReuseType::NUMERIC_REUSE_BASE ||
        Options().getNumericReuseType() ==
            NumericReuseType::NUMERIC_REUSE_PARALLEL) {
      if (contact_points.empty()) {
        if (chol_L_prev != nullptr) {
          chol_L = chol_L_prev;
        }
        copy_prev_factor = false;
        return;
      }
    }

    regions.assignMeshPtr(M_n, Mp, Mi);
    regions.getGlobalPerm(contact_points, mesh_perm);
#pragma omp parallel for // TODO: only fix the region with change
    for (int idx = 0; idx < M_n; idx++) {
      hessian_perm[idx * 3] = mesh_perm[idx] * 3;
      hessian_perm[idx * 3 + 1] = mesh_perm[idx] * 3 + 1;
      hessian_perm[idx * 3 + 2] = mesh_perm[idx] * 3 + 2;
    }
  }

#ifndef NDEBUG // TODO: DELETE this
  std::vector<bool> perm_flag(A_n, false);
  for (int i = 0; i < A_n; i++) {
    perm_flag[hessian_perm[i]] = true;
  }
  double invalid_perm = 0;
  for (int i = 0; i < A_n; i++) {
    if (!perm_flag[i]) {
      invalid_perm++;
      assert(false);
    }
  }
  if (invalid_perm != 0) {
    std::cerr << "Your permutation doesn't cover: " << invalid_perm / A_n
              << std::endl;
    assert(invalid_perm == 0);
  }
#endif
  // Apply the reordering and perform the analysis
  A_cm.nmethods = 1;
  A_cm.method[0].ordering = CHOLMOD_GIVEN;
  if (Options().getNumericReuseType() == NumericReuseType::NUMERIC_REUSE_BASE ||
      Options().getNumericReuseType() == NumericReuseType::NUMERIC_NO_REUSE) {
    A_cm.postorder = 1;
  } else {
    A_cm.postorder = 1;
  }
  A_cm.supernodal = CHOLMOD_SUPERNODAL;
  std::cout << "+++PARTH: Number of contact points: "
            << regions.num_contact_points << std::endl;
  assert(chol_L == nullptr);
  chol_L =
      cholmod_analyze_p_custom(chol_A, hessian_perm.data(), NULL, 0, &A_cm);
  if (chol_L->is_super != 1) {
    std::cerr << "***PARTH: THE FACTORIZATION IS NOW SIMPLICAL - everything is "
                 "probably won't work"
              << std::endl;
    Options().setNumericReuseType(NumericReuseType::NUMERIC_NO_REUSE);
    // TODO: CREATE SOMETHING FOR SWITCHING BACK
  }
  assert(chol_L != nullptr);

  tree_is_computed = false;

  if (Options().getNumericReuseType() ==
      NumericReuseType::NUMERIC_NO_REUSE_PARALLEL) {
    //    createParallelSchedule();
    createSpTRSVParallelSchedule();
  } else if (Options().getNumericReuseType() ==
             NumericReuseType::NUMERIC_REUSE_PARALLEL) {
    createSpTRSVParallelSchedule();
    //    createReuseParallelSchedule();
    //    createParallelSchedule();
  }
  assert(chol_L != nullptr);
  total_num_of_super = chol_L->nsuper;
}

void ParthSolver::analyze(bool do_analysis) {
  double start = omp_get_wtime();
  std::cout << "+++ PARTH: Analysis +++" << std::endl;
  if (chol_L != nullptr) {
    cholmod_free_factor(&chol_L, &A_cm);
  }

  if (Options().getNumericReuseType() == NumericReuseType::NUMERIC_NO_REUSE ||
      Options().getNumericReuseType() ==
          NumericReuseType::NUMERIC_NO_REUSE_PARALLEL) {
    do_analysis = true;
  }

  switch (Options().getAnalyzeType()) {
  case AnalyzeType::Hessian:
    this->hessianAnalyze();
    break;
  case AnalyzeType::Mesh:
    if (this->mesh_init) {
      this->meshAnalyze();
    } else {
      std::cerr << "*** PARTH: please initialize the mesh first" << std::endl;
      return;
    }
    break;
  case AnalyzeType::SimulationAware:
    if (!do_analysis) {
      analyze_is_requested = true;
      return;
    }
    analyze_is_requested = false;
    this->SimulationAwareAnalyze();
    break;
  default:
    std::cerr << "*** PARTH: This analysis type is not define" << std::endl;
    return;
  }
  M_n_prev = M_n;
  M_nnz_prev = Mp[M_n];

  Mp_prev.resize(M_n_prev + 1);
  Mi_prev.resize(M_nnz_prev);
  std::copy(Mp, Mp + M_n + 1, Mp_prev.data());
  std::copy(Mi, Mi + M_nnz_prev, Mi_prev.data());
  //  if (opt.getVerbose()) {
  //    std::cout << "*** PARTH: Number of Nodes: " << M_n << std::endl;
  //    std::cout << "*** PARTH: Number of NNZ: " << chol_A->nzmax << std::endl;
  //    std::cout << "*** PARTH: Number of L non-zeros: "
  //              << this->getFactorNonzeros() << std::endl;
  //    std::cout << "*** PARTH: Number of L non-zeros / A non-zeros: "
  //              << this->getFactorNonzeros() / chol_A->nzmax << std::endl;
  //  }
  analyze_total_time = omp_get_wtime() - start;
}

std::vector<int> &ParthSolver::getElemRegions() {
  if (regions.nodes_regions.empty()) {
    regions.nodes_regions.resize(M_n, 1);
  }
  return regions.nodes_regions;
}

void ParthSolver::computeContacts(std::vector<int> &contact_points) {
  contact_points.clear();
  if (M_n_prev != M_n) {
    contact_points.resize(M_n);
    for (int i = 0; i < M_n; i++) {
      contact_points[i] = i;
    }
    return;
  }

  std::vector<int> Mp_tmp(M_n + 1);
  std::copy(Mp, Mp + M_n + 1, Mp_tmp.data());
  std::vector<int> Mi_tmp(Mp[M_n]);
  std::copy(Mi, Mi + Mp[M_n], Mi_tmp.data());

  for (int i = 0; i < M_n; i++) {
    int num_nbr = Mp[i + 1] - Mp[i];
    int num_nbr_prev = Mp_prev[i + 1] - Mp_prev[i];
    if (num_nbr != num_nbr_prev) {
      contact_points.emplace_back(i);
      continue;
    }
    int prev_cnt = Mp_prev[i];
    int cnt = Mp[i];
    assert(num_nbr == num_nbr_prev);
    for (int j = 0; j < num_nbr; j++) {
      if (Mi[cnt++] != Mi_prev[prev_cnt++]) {
        contact_points.emplace_back(i);
        break;
      }
    }
  }
}

void ParthSolver::createParallelSchedule() {
  // Create the coarsend partitions
  int nsuper = chol_L->nsuper;
  auto super = (int *)chol_L->super;
  int n_elem = chol_L->n / 3;
  assert(chol_L->n % 3 == 0);

  std::vector<int> element_to_super(n_elem);
  for (int s = 0; s < nsuper; s++) {
    int start_elem = super[s] / 3;
    int end_elem = super[s + 1] / 3;
    for (int e = start_elem; e < end_elem; e++) {
      element_to_super[e] = s;
    }
  }

  // Start from a level with enough regions -> Note that levels are start from
  // 0
  // So parallel_levels equal to 1 means we start with level 0 with one region
  // and level 1 with 2 regions
  parallel_levels = std::min(std::ceil(std::log2(Options().getNumberOfCores())),
                             (double)regions.num_levels);

  //-------------------------------------------------
  // The structure is as follow:
  // parallel_levels: The number of levels that we used from the region
  // tree for region parallelization
  //
  // level_ptr: is the ptr to region_ptr array that shows the supernodes
  // belong to each region region_ptr: is a ptr to supernode_idx array that
  // shows the supernodes in each reigon
  //-------------------------------------------------

  level_ptr.clear();
  part_ptr.clear();
  supernode_idx.clear();
  // Exp:|0|#regions in lvl1|#regions in lvl2 + lvl1|
  level_ptr.reserve(parallel_levels + 1);
  // total #regions -> we may have empty regions for separated meshes so we
  // reserve
  part_ptr.reserve(std::pow(2, parallel_levels + 1));
  supernode_idx.reserve(nsuper);

  level_ptr.emplace_back(0);
  part_ptr.emplace_back(0);
  std::vector<int> part_max_size;
  std::vector<int> part_max_super;
  std::vector<int> part_lvl;

  int switch_th = 1024;
  std::vector<int> flag(nsuper, 0);
  std::vector<int> serial_lvl;
  for (int l = parallel_levels; l >= 0; l--) {
    int start_region = std::pow(2, l) - 1;
    int end_region = std::pow(2, l) + start_region;
    int part_cnt = 0;
    serial_lvl.clear();
    for (int r = start_region; r < end_region; r++) {
      int start_col = 0;
      int end_col = 0;
      double max_region_size = 0;
      int max_super_id = 0;

      if (l == parallel_levels) {
        start_col = regions.getSubTreeOffset(regions.tree_nodes[r]) * 3;
        end_col = start_col + regions.getSubTreeSize(regions.tree_nodes[r]) * 3;
      } else {
        start_col = regions.tree_nodes[r].offset * 3;
        end_col = start_col + regions.tree_nodes[r].assigned_nodes.size() * 3;
      }
      // empty regions are skipped.
      if (start_col == end_col) {
        continue;
      }

      // Adding supernodes of this part to the parallel and serial levels
      while (start_col < end_col) {
        int elem = start_col / 3;
        int super_idx = element_to_super[elem];

        // If the supernode is already handles (it was a parent of a serial
        // supernode)
        if (flag[super_idx] == 1) {
          start_col = super[super_idx + 1];
          continue;
        }

        // Dealing with big supernodes that should be run using supernode
        // parallelism (cholmod default)
        if (super_nz[super_idx] > switch_th) {
          //        if (super[super_idx] -
          //        first_nonzero_per_row[super[super_idx]] >
          //            switch_th) {
          serial_lvl.emplace_back(super_idx);
          flag[super_idx] = 1;
          int parent = super_parents[super_idx];
          while (parent != -1) {
            // If super is in the same region and  we didn't reach to this
            // parent from any other supernode
            if (super[parent + 1] - 1 < end_col && flag[parent] != 1) {
              serial_lvl.emplace_back(parent);
              flag[parent] = 1;
              parent = super_parents[parent];
            } else {
              break;
            }
          }
          start_col = super[super_idx + 1];
          continue;
        }

        // The small supernodes that can be run in parallel in node
        // parallelism matter
        supernode_idx.emplace_back(super_idx);
        flag[super_idx] = 1;
        assert(super[super_idx + 1] - 1 < end_col);
        if (max_region_size < super_nz[super_idx]) {
          max_region_size = super_nz[super_idx];
          max_super_id = super_idx;
        }
        start_col = super[super_idx + 1];
      }
      if (part_ptr.back() != supernode_idx.size()) {
        part_ptr.emplace_back(supernode_idx.size());
        part_cnt++;
      }
    }

    // Adding the two levels + serial part into the stack
    int back = level_ptr.back();
    if (level_ptr.back() != part_cnt + back) {
      level_ptr.emplace_back(part_cnt + back);
    }

    // TODO: IT IS VERY INEFFICIENT
    if (!serial_lvl.empty()) {
      int level_last = level_ptr.size() - 1;
      int num_parts_prev = level_ptr[level_last] - level_ptr[level_last - 1];
      if (num_parts_prev == 1) {
        int start_supernode_idx = part_ptr[level_ptr[level_last - 1]];
        supernode_idx.insert(supernode_idx.end(), serial_lvl.begin(),
                             serial_lvl.end());
        // If we reach to a parent from two different nodes, we should sort
        // for dependencies
        std::sort(supernode_idx.data() + start_supernode_idx,
                  supernode_idx.data() + supernode_idx.size());
        part_ptr.back() = supernode_idx.size();
      } else {
        // If we reach to a parent from two different nodes, we should sort
        // for dependencies
        std::sort(serial_lvl.begin(), serial_lvl.end());
        supernode_idx.insert(supernode_idx.end(), serial_lvl.begin(),
                             serial_lvl.end());
        part_ptr.emplace_back(supernode_idx.size());
        level_ptr.emplace_back(part_ptr.size() - 1);
      }
    }
  }
  assert(supernode_idx.size() == nsuper);

#ifndef NDEBUG
  part_max_size.reserve(part_ptr.capacity());
  part_lvl.reserve(part_ptr.capacity());
  part_max_super.reserve(part_ptr.capacity());
  for (int l = 0; l < level_ptr.size() - 1; l++) {
    for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
      int max_part_size = 0;
      int max_part_super_id = 0;
      for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
        int s = supernode_idx[s_ptr];
        if (max_part_size < super_nz[s]) {
          max_part_size = super_nz[s];
          max_part_super_id = s;
        }
      }
      part_max_size.emplace_back(max_part_size);
      part_lvl.emplace_back(l);
      part_max_super.emplace_back(max_part_super_id);
    }
  }

  {
    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("super_id");
    Runtime_headers.emplace_back("nz");

    PARTH::CSVManager runtime_csv(
        "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/"
        "superNode_Update_" +
            std::to_string(frame) + "_" + std::to_string(iter),
        "some address", Runtime_headers, false);

    for (int s = 0; s < nsuper; s++) {
      runtime_csv.addElementToRecord(s, "super_id");
      runtime_csv.addElementToRecord(super_nz[s], "nz");
      runtime_csv.addRecord();
    }
  };

  {
    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("Part_id");
    Runtime_headers.emplace_back("Level_id");
    Runtime_headers.emplace_back("Super_id");
    Runtime_headers.emplace_back("nz");

    PARTH::CSVManager runtime_csv(
        "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/"
        "Part_Update_" +
            std::to_string(frame) + "_" + std::to_string(iter),
        "some address", Runtime_headers, false);

    for (int p = 0; p < part_max_size.size(); p++) {
      runtime_csv.addElementToRecord(p, "Part_id");
      runtime_csv.addElementToRecord(part_lvl[p], "Level_id");
      runtime_csv.addElementToRecord(part_max_super[p], "Super_id");
      runtime_csv.addElementToRecord(part_max_size[p], "nz");
      runtime_csv.addRecord();
    }
  }

  std::vector<int> super_to_lvl(nsuper, -1);
  std::vector<int> super_to_part(nsuper, -1);
  // Applying Checks for correct dependencies
  for (int l = 0; l < level_ptr.size() - 1; l++) {
    for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
      for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
        int s = supernode_idx[s_ptr];
        super_to_lvl[s] = l;
        super_to_part[s] = p;
      }
    }
  }

  // Applying Checks for correct dependencies
  for (int l = 0; l < level_ptr.size() - 1; l++) {
    for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
      int prev_s = supernode_idx[part_ptr[p]];
      for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
        int s = supernode_idx[s_ptr];
        int parent = super_parents[s];
        if (parent == -1) {
          continue;
        }
        if (super_to_lvl[parent] == l) {
          assert(super_to_part[parent] == p);
        } else if (super_to_lvl[parent] > l) {
          assert(super_to_part[parent] != p);
        } else {
          assert(false);
        }
      }
    }
  }

  assert(supernode_idx.size() == nsuper);
  assert(part_ptr.size() - 1 == level_ptr.back());
  std::vector<int> marked(nsuper, 0);
  for (int l = 0; l < level_ptr.size() - 1; l++) {
    for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
      for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
        marked[supernode_idx[s_ptr]] = 1;
      }
    }
  }

  for (auto iter : marked) {
    if (iter == 0) {
      assert(false);
    }
  }
#endif
}

// Comparison function to sort the vector elements
// by second element of tuples
bool sortbysec(const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
  return (std::get<1>(a) > std::get<1>(b));
}

void ParthSolver::createSpTRSVParallelSchedule() {
  // Compute Etree
  // Create the coarsend partitions
  int nsuper = chol_L->nsuper;
  auto super = (int *)chol_L->super;
  int n_elem = chol_L->n / 3;
  assert(chol_L->n % 3 == 0);

  //  Creating the inverse elemination tree
  computeInverseEliminationTree(nsuper, super_parents.data(), tree_ptr,
                                tree_set);

  int switch_th = 256;
  std::vector<int> serial_nodes(nsuper, 0);
  bool serial_node_exist = false;
  // Define serial nodes //TODO: MAKE IT MORE EFFICIENT
  for (int s = 0; s < nsuper; s++) {
    if (super[s + 1] - super[s] > switch_th) {
      serial_nodes[s] = 1;
      serial_node_exist = true;
      int parent = super_parents[s];
      while (parent != -1) {
        serial_nodes[parent] = 1;
        parent = super_parents[parent];
      }
    }
  }

  // Find the first cut based on children of serial nodes that are not serial
  std::vector<std::tuple<int, int>> first_cut;
  if (serial_node_exist) {
    for (int s = 0; s < nsuper; s++) {
      if (serial_nodes[s] == 1) {
        for (int child_ptr = tree_ptr[s]; child_ptr < tree_ptr[s + 1];
             child_ptr++) {
          int child = tree_set[child_ptr];
          if (serial_nodes[child] == 0) {
            first_cut.emplace_back(
                std::tuple<int, int>(child, super[child + 1] - super[child]));
          }
        }
      }
    }
  }

  // If there was no serial nodes, find the roots
  for (int s = 0; s < nsuper; s++) {
    if (super_parents[s] == -1 && serial_nodes[s] == 0) {
      first_cut.emplace_back(std::tuple<int, int>(s, super[s + 1] - super[s]));
    }
  }
  assert(first_cut.size() != 0);

  while (first_cut.size() < Options().getNumberOfCores()) {
    std::sort(first_cut.begin(), first_cut.end(), sortbysec);
    int node = std::get<0>(first_cut[0]);
    assert(serial_nodes[node] == 0);
    serial_nodes[node] = 1;
    first_cut[0] = first_cut.back();
    first_cut.pop_back();

    for (int child_ptr = tree_ptr[node]; child_ptr < tree_ptr[node + 1];
         child_ptr++) {
      int child = tree_set[child_ptr];
      if (serial_nodes[child] != 1) {
        first_cut.emplace_back(
            std::tuple<int, int>(child, super[child + 1] - super[child]));
      }
    }
  }

#ifndef NDEBUG
  std::vector<int> first_cut_mark(nsuper, 0);
  for (auto &iter : first_cut) {
    first_cut_mark[std::get<0>(iter)] = 1;
  }
  for (auto &f : first_cut) {
    int s = std::get<0>(f);
    assert(serial_nodes[s] == 0);
    int parent = super_parents[s];
    while (parent != -1) {
      assert(serial_nodes[parent] == 1);
      assert(first_cut_mark[parent] == 0);
      parent = super_parents[parent];
    }
  }
#endif

  //-------------------------------------------------
  // The structure is as follow:
  // parallel_levels: The number of levels that we used from the region
  // tree for region parallelization
  //
  // level_ptr: is the ptr to region_ptr array that shows the supernodes
  // belong to each region region_ptr: is a ptr to supernode_idx array that
  // shows the supernodes in each reigon
  //-------------------------------------------------

  level_ptr.clear();
  part_ptr.clear();
  supernode_idx.clear();
  // Exp:|0|#regions in lvl1|#regions in lvl2 + lvl1|
  parallel_levels = 2;
  level_ptr.reserve(parallel_levels + 1);
  // total #regions -> we may have empty regions for separated meshes so we
  // reserve
  part_ptr.reserve(first_cut.size());
  supernode_idx.reserve(nsuper);

  level_ptr.emplace_back(0);
  part_ptr.emplace_back(0);

  // Sort based on node idx
  std::sort(first_cut.begin(), first_cut.end());
  int num_parts = 0;
  //  std::vector<int> s_to_p(nsuper, -1);
  // Add parallel part
  for (auto &f : first_cut) {
    // Mark all the descendent
    std::queue<int> r_queue;
    r_queue.push(std::get<0>(f));
    while (!r_queue.empty()) {
      int node = r_queue.front();
      r_queue.pop();
      assert(serial_nodes[node] == 0);
      supernode_idx.emplace_back(node);
      //      s_to_p[node] = part_ptr.size() - 1;
      //      if (serial_nodes[super_parents[node]] == 0) {
      //        assert(s_to_p[super_parents[node]] == s_to_p[node]);
      //      }
      for (int child_ptr = tree_ptr[node]; child_ptr < tree_ptr[node + 1];
           child_ptr++) {
        int child = tree_set[child_ptr];
        assert(serial_nodes[child] == 0);
        r_queue.push(child);
      }
    }
    std::sort(supernode_idx.data() + part_ptr.back(),
              supernode_idx.data() + supernode_idx.size());
    part_ptr.emplace_back(supernode_idx.size());
  }
  level_ptr.emplace_back(part_ptr.size() - 1);

#ifndef NDEBUG
  for (int s = 0; s < supernode_idx.size(); s++) {
    assert(serial_nodes[supernode_idx[s]] == 0);
  }
#endif
  // Add serial part
  for (int s = 0; s < nsuper; s++) {
    if (serial_nodes[s] == 1) {
      supernode_idx.emplace_back(s);
    }
  }
  std::sort(supernode_idx.data() + part_ptr.back(),
            supernode_idx.data() + supernode_idx.size());
  part_ptr.emplace_back(supernode_idx.size());
  level_ptr.emplace_back(part_ptr.size() - 1);
  assert(supernode_idx.size() == nsuper);
#ifndef NDEBUG
  // check data-dependency
  std::vector<int> super_to_lvl(nsuper, -1);
  std::vector<int> super_to_part(nsuper, -1);
  // Applying Checks for correct dependencies
  for (int l = 0; l < level_ptr.size() - 1; l++) {
    for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
      for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
        int s = supernode_idx[s_ptr];
        super_to_lvl[s] = l;
        super_to_part[s] = p;
      }
    }
  }

  // Applying Checks for correct dependencies
  for (int l = 0; l < level_ptr.size() - 1; l++) {
    for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
      int prev_s = supernode_idx[part_ptr[p]];
      for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
        int s = supernode_idx[s_ptr];
        int parent = super_parents[s];
        if (parent == -1) {
          continue;
        }
        if (super_to_lvl[parent] == l) {
          assert(super_to_part[parent] == p);
        } else if (super_to_lvl[parent] > l) {
          assert(super_to_part[parent] != p);
        } else {
          assert(false);
        }
      }
    }
  }

  // Check existance
  std::vector<int> marked(nsuper, 0);
  for (auto &iter : supernode_idx) {
    marked[iter] = 1;
  }
  for (auto &iter : marked) {
    if (iter == 0) {
      assert(false);
    }
  }
#endif
}

int ParthSolver::computeETreeCost(int *super, int *tree_ptr, int *tree_set,
                                  int current_node) {
  assert(tree_ptr != nullptr);
  assert(tree_set != nullptr);
  assert(super != nullptr);
  assert(current_node > -1);
  int child_cost = 0;
  for (int child_ptr = tree_ptr[current_node];
       child_ptr < tree_ptr[current_node + 1]; child_ptr++) {
    int child = tree_set[child_ptr];
    child_cost += computeETreeCost(super, tree_ptr, tree_set, child);
  }
  node_cost[current_node] =
      child_cost + super[current_node + 1] - super[current_node];
}

void ParthSolver::createReuseParallelSchedule() {
  // Create the coarsend partitions
  int nsuper = chol_L->nsuper;
  auto super = (int *)chol_L->super;
  int n_elem = chol_L->n / 3;
  assert(chol_L->n % 3 == 0);

  // A cost function that works on Bud hava
  // Mark the serial supernodes and all of their parents
  std::vector<int> serial_Marked(nsuper, -1);
  int switch_th = 256;
  std::vector<int> serial_supernodes_id;
  for (int s = 0; s < nsuper; s++) {
    if (serial_Marked[s] == 1) {
      continue;
    }
    int max_distance = 0;
    for (int k = super[s]; k < super[s + 1]; k++) {
      if (k - first_nonzero_per_row[k] > max_distance) {
        max_distance = k - first_nonzero_per_row[k];
      }
    }
    if (max_distance > switch_th) {
      serial_Marked[s] = 1;
      serial_supernodes_id.emplace_back(s);
      int parent = super_parents[s];
      while (parent != -1) {
        serial_Marked[parent] = 1;
        serial_supernodes_id.emplace_back(parent);
        parent = super_parents[parent];
      }
    }
  }

  if (Options().getVerbose()) {
    std::cout << "+++PARTH: The percentage of serial supernodes are: "
              << serial_supernodes_id.size() * 1.0 / nsuper << std::endl;
  }

  // Compute the cost of computation based on the size of the supernodes
  computeInverseEliminationTree(nsuper, super_parents.data(), tree_ptr,
                                tree_set);

  std::vector<int> roots;
  for (int s = 0; s < nsuper; s++) {
    if (super_parents[s] == -1) {
      roots.emplace_back(s);
    }
  }

  node_cost.clear();
  node_cost.resize(nsuper, 0);

  for (auto &r : roots) {
    computeETreeCost(super, tree_ptr.data(), tree_set.data(), r);
  }

  // Start from a level with enough regions -> Note that levels are start from
  // 0
  // So parallel_levels equal to 1 means we start with level 0 with one region
  // and level 1 with 2 regions
  parallel_levels = std::min(std::ceil(std::log2(Options().getNumberOfCores())),
                             (double)regions.num_levels);

  //-------------------------------------------------
  // The structure is as follow:
  // parallel_levels: The number of levels that we used from the region
  // tree for region parallelization
  //
  // level_ptr: is the ptr to region_ptr array that shows the supernodes
  // belong to each region region_ptr: is a ptr to supernode_idx array that
  // shows the supernodes in each reigon
  //-------------------------------------------------

  level_ptr.clear();
  part_ptr.clear();
  supernode_idx.clear();
  // Exp:|0|#regions in lvl1|#regions in lvl2 + lvl1|
  level_ptr.reserve(parallel_levels + 1);
  // total #regions -> we may have empty regions for separated meshes so we
  // reserve
  part_ptr.reserve(std::pow(2, parallel_levels + 1));
  supernode_idx.reserve(nsuper);

  level_ptr.emplace_back(0);
  part_ptr.emplace_back(0);

  // Find a cut with good PG

  // Create the schedule
}

void ParthSolver::computeFirstchild(int n, int *Ap, int *Ai) {
  first_nonzero_per_row.clear();
  first_nonzero_per_row.resize(n, n);
  for (int c = 0; c < n; c++) {
    first_nonzero_per_row[c] = Ai[Ap[c]];
  }
}

} // namespace PARTH
