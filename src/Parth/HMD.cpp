//
// Created by behrooz zare on 2024-04-07.
//

#include "HMD.h"
#include <suitesparse/amd.h>
#include <cmath>
#include <iostream>
#include <metis.h>
#include <queue>

namespace PARTH {

HMD::~HMD() {
//    delete p2pScheduleForDecomposing;
}
void HMD::clearHMD() {
  verbose = true; // Verbose mode for debugging
  PARTH::ReorderingType reorder_type = PARTH::ReorderingType::METIS;

  HMD_tree.clear();
  HMD_init = false;
  num_levels = 0;
  num_HMD_nodes = 0;
  DOF_to_HMD_node.clear();
}

int HMD::getLastHMDNodeInLevel(int level) const {
  int node = std::pow(2, level + 1) - 1;
  assert(node < num_HMD_nodes);
  return node;
}

void HMD::metisNodeNDPermutation(int M_n, int *Mp, int *Mi, int *perm,
                                 int *Iperm) {
  idx_t N = M_n;
  idx_t NNZ = Mp[M_n];
  if (NNZ == 0) {
    assert(M_n != 0);
    for (int i = 0; i < M_n; i++) {
#ifndef NDEBUG
//      std::cout << "WARNING: This decomposition does not have edges"
//                << std::endl;
#endif
      perm[i] = i;
    }
    return;
  }
  // TODO add memory allocation protection later like CHOLMOD
  if (Iperm == nullptr) {
    std::vector<int> tmp(M_n);
    METIS_NodeND(&N, Mp, Mi, NULL, NULL, perm, tmp.data());
  } else {
    METIS_NodeND(&N, Mp, Mi, NULL, NULL, perm, Iperm);
  }
}

void HMD::AMDNodeNDPermutation(int M_n, int *Mp, int *Mi, int *perm,
                               int *Iperm) {
  idx_t N = M_n;
  idx_t NNZ = Mp[M_n];
  if (NNZ == 0) {
    assert(M_n != 0);
    for (int i = 0; i < M_n; i++) {
#ifndef NDEBUG
//      std::cout << "WARNING: This decomposition does not have edges"
//                << std::endl;
#endif
      perm[i] = i;
    }
    return;
  }
  // TODO add memory allocation protection later like CHOLMOD
  amd_order(N, Mp, Mi, perm, nullptr, nullptr);
}

void HMD::Permute(int M_n, int *Mp, int *Mi, int *perm, int *Iperm) {
  if (this->reorder_type == PARTH::ReorderingType::METIS) {
    metisNodeNDPermutation(M_n, Mp, Mi, perm, Iperm);
  } else if (this->reorder_type == PARTH::ReorderingType::AMD) {
    AMDNodeNDPermutation(M_n, Mp, Mi, perm, Iperm);
  } else if (this->reorder_type == PARTH::ReorderingType::MORTON_CODE) {
    std::cerr << "Morton code is not implemented yet" << std::endl;
  } else {
    std::cerr << "Reordering type is not defined" << std::endl;
  }
}

void HMD::initHMD(
    int num_levels, /// <[in] How many levels you want to dissect the graph
    int M_n,        ///<[in] Number of elements inside the mesh
    int *Mp,        ///<[in] Pointer array of mesh in CSC format
    int *Mi,        ///<[in] Index array of mesh in CSC format
    std::vector<int> &mesh_perm ///<[out] The mesh permutation array
) {

  assert(num_levels >= 0);
  if (num_levels > 12) { // What
    num_levels = 10;
  }
  this->num_levels = num_levels;
  this->num_HMD_nodes = std::pow(2, num_levels + 1) - 1;
  HMD_tree.clear();
  this->HMD_tree.resize(num_HMD_nodes);
  DOF_to_HMD_node.clear();
  this->DOF_to_HMD_node.resize(M_n, 0);

  std::vector<int> assigned_nodes(M_n);
  for (int i = 0; i < M_n; i++) {
    assigned_nodes[i] = i;
  }

//  computeP2PSchedule(0, num_HMD_nodes);
  this->Decompose(0, -1, 0, M_n, Mp, Mi, assigned_nodes, 0);

  mesh_perm.resize(M_n);
  assignCoarseLocalPermToGlobal(HMD_tree[0], mesh_perm);

  this->HMD_init = true;
}

bool HMD::Decompose(
    int HMD_node_id,   ///[in] HMD node id
    int HMD_parent_id, /// <[in] HMD node's parent id
    int cur_level,     ///<[in] The current level that we are dissecting
    int M_n,           ///<[in] Number of nodes in the mesh
    int *Mp,           ///<[in] The pointer array of mesh
    int *Mi,           ///<[in] The index array of mesh
    std::vector<int> &
        HMD_node_DOFs, ///<[in] assigned DOF to this HMD node and its descendent
    int offset         ///<[in] number of nodes before this region
) {
  assert(M_n != 0);

  auto &cur_node = this->HMD_tree[HMD_node_id];
  //+++++++++++++ Boundary condition of a single level ++++++++++++++++
  if (cur_level == num_levels) {
    // Init the assigned nodes
    cur_node.DOFs = HMD_node_DOFs;
    cur_node.setIdentifier(-1, -1, HMD_node_id, HMD_parent_id, offset,
                           cur_level, cur_node.DOFs.size());
    cur_node.setInitFlag();

    // Assign the region to the global region
    for (auto &m_node : HMD_node_DOFs) {
      this->DOF_to_HMD_node[m_node] = HMD_node_id;
    }

    // Assign the permuted labels
    std::vector<int> perm(M_n);
    Permute(M_n, Mp, Mi, perm.data(), nullptr);
    cur_node.setPermutedNewLabel(perm);
    return true;
  }

  // Prepare the Wavefront parallelism arrays and data
  int total_number_of_sub_tree_nodes =
      std::pow(2, (num_levels - cur_level) + 1) - 1;
  // Levels are zero based. So for num_levels = 1 we have two levels 0 and 1,
  // and we need an array of size 3 to define wavefront parallelism
  int wavefront_levels = num_levels - cur_level + 1;

  std::vector<int> level_ptr(wavefront_levels + 1);
  std::vector<int> tree_node_ids_set(total_number_of_sub_tree_nodes);
  std::vector<int> tree_node_ids_set_inv(this->num_HMD_nodes);
  std::vector<int> parent_node_ids_set(total_number_of_sub_tree_nodes);
  // These variables should be computed on the fly
  std::vector<int> offset_set(total_number_of_sub_tree_nodes);
  std::vector<std::vector<int>> sub_mesh_assigned_nodes(
      total_number_of_sub_tree_nodes);
  std::vector<SubMesh> sub_mesh_stack(total_number_of_sub_tree_nodes);

  int size_per_level = 1;
  level_ptr[0] = 0;
  for (int l = 1; l < wavefront_levels + 1; l++) {
    level_ptr[l] = size_per_level + level_ptr[l - 1];
    size_per_level = size_per_level * 2;
  }

  tree_node_ids_set[0] = HMD_node_id;
  parent_node_ids_set[0] = HMD_parent_id;
  for (int l = 0; l < wavefront_levels - 1; l++) {
    int next_level_idx = level_ptr[l + 1];
    for (int node_ptr = level_ptr[l]; node_ptr < level_ptr[l + 1]; node_ptr++) {
      int node_idx = tree_node_ids_set[node_ptr];
      // Left node
      parent_node_ids_set[next_level_idx] = node_idx;
      tree_node_ids_set[next_level_idx++] = node_idx * 2 + 1;
      // Right node
      parent_node_ids_set[next_level_idx] = node_idx;
      tree_node_ids_set[next_level_idx++] = node_idx * 2 + 2;
    }
  }

  for (int i = 0; i < total_number_of_sub_tree_nodes; i++) {
    tree_node_ids_set_inv[tree_node_ids_set[i]] = i;
  }

  // TODO: Prune the mesh
  sub_mesh_stack[0].M_n = M_n;
  sub_mesh_stack[0].M_nnz = Mp[M_n];
  sub_mesh_stack[0].Mp.resize(M_n + 1);
  sub_mesh_stack[0].Mi.resize(Mp[M_n]);
  std::copy(Mp, Mp + M_n + 1, sub_mesh_stack[0].Mp.data());
  std::copy(Mi, Mi + Mp[M_n], sub_mesh_stack[0].Mi.data());
  sub_mesh_assigned_nodes[0] = HMD_node_DOFs;
  offset_set[0] = offset;



//NOTE: you SHOULD comment these pragmas if you want to re-generate the Parth results for the paper
//In the paper, the results were NOT for parallel execution of decomposition and ordering
// start the wavefront parallelism 
#pragma omp parallel
  {
    for (int l = 0; l < wavefront_levels; l++) {
#pragma omp for schedule(dynamic)
      for (int node_ptr = level_ptr[l]; node_ptr < level_ptr[l + 1];
           node_ptr++) {

        const int &id = tree_node_ids_set[node_ptr];
        const int &parent_id = parent_node_ids_set[node_ptr];
        const int &current_level = l + cur_level;
        auto &mesh = sub_mesh_stack[node_ptr];
        const auto &assigned_nodes_par = sub_mesh_assigned_nodes[node_ptr];
        const auto &offset_par = offset_set[node_ptr];
        if (mesh.M_n == 0) {
          continue;
        }
        int M_n = mesh.M_n;
        int *Mp = mesh.Mp.data();
        int *Mi = mesh.Mi.data();
        auto &current_node = this->HMD_tree[id];
        //+++++++++++++ Boundary condition of a single level ++++++++++++++++
        if (current_level == num_levels || mesh.M_nnz == 0) {
          current_node.setIdentifier(-1, -1, id, parent_id, offset_par,
                                     current_level, assigned_nodes_par.size());
          // Init the assigned nodes
          current_node.DOFs = assigned_nodes_par;
          current_node.setInitFlag();

          // Assign the region to the global region
          for (auto &m_node : assigned_nodes_par) {
            this->DOF_to_HMD_node[m_node] = id;
          }

          // Assign the permuted labels to the sep nodes
          std::vector<int> perm(M_n);
          Permute(M_n, Mp, Mi, perm.data(), nullptr);
          current_node.setPermutedNewLabel(perm);
          continue;
        }

        std::vector<int> local_nodes_regions(M_n);
        idx_t nVertices = M_n;
        idx_t csp;
        std::vector<int> vweight(nVertices, 1);
        int ret = METIS_ComputeVertexSeparator(&nVertices, Mp, Mi,
                                               vweight.data(), NULL, &csp,
                                               local_nodes_regions.data());

        if (ret != METIS_OK) {
          std::cerr << "Something went wrong" << std::endl;
        }

        auto &left_region = sub_mesh_stack[tree_node_ids_set_inv[id * 2 + 1]];
        auto &right_region = sub_mesh_stack[tree_node_ids_set_inv[id * 2 + 2]];
        SubMesh sep_region;

        createTriSubMesh(local_nodes_regions, M_n, Mp, Mi, left_region,
                         right_region, sep_region);

        // Assign nodes
        auto &left_assigned =
            sub_mesh_assigned_nodes[tree_node_ids_set_inv[id * 2 + 1]];
        auto &right_assigned =
            sub_mesh_assigned_nodes[tree_node_ids_set_inv[id * 2 + 2]];
        auto &sep_assigned = current_node.DOFs;
        left_assigned.reserve(left_region.M_n);
        right_assigned.reserve(right_region.M_n);
        sep_assigned.reserve(sep_region.M_n);

        for (int i = 0; i < local_nodes_regions.size(); i++) {
          if (local_nodes_regions[i] == 0) { // Left assigned
            left_assigned.emplace_back(assigned_nodes_par[i]);
          } else if (local_nodes_regions[i] == 1) {
            right_assigned.emplace_back(assigned_nodes_par[i]);
          } else if (local_nodes_regions[i] == 2) {
            sep_assigned.emplace_back(assigned_nodes_par[i]);
          } else {
            std::cerr << "There are more than 3 regions in here" << std::endl;
          }
        }

        // Assign the permuted labels to the sep nodes
        sep_region.perm.resize(sep_region.M_n);
        if (sep_region.M_n != 0) {
          Permute(sep_region.M_n, sep_region.Mp.data(), sep_region.Mi.data(),
                  sep_region.perm.data(), nullptr);
        }

        if (left_region.M_n == 0 && right_region.M_n == 0) {
          current_node.setIdentifier(-1, -1, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        } else if (left_region.M_n == 0) {
          current_node.setIdentifier(-1, id * 2 + 2, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        } else if (right_region.M_n == 0) {
          current_node.setIdentifier(id * 2 + 1, -1, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        } else {
          current_node.setIdentifier(id * 2 + 1, id * 2 + 2, id, parent_id,
                                     offset_par + left_region.M_n +
                                         right_region.M_n,
                                     current_level, current_node.DOFs.size());
        }

        current_node.setInitFlag();
        current_node.setPermutedNewLabel(sep_region.perm);

        // assign the regions to the global node regions
        for (auto &sep_node : sep_assigned) {
          this->DOF_to_HMD_node[sep_node] = id;
        }

        // Left offset
        offset_set[tree_node_ids_set_inv[id * 2 + 1]] = offset_par;
        // Right offset
        offset_set[tree_node_ids_set_inv[id * 2 + 2]] =
            offset_par + left_region.M_n;

        // Clear the vectors values to release memory
        sub_mesh_stack[node_ptr].clear();
        sub_mesh_assigned_nodes[node_ptr].clear();
      }
    }
  };
  return true;
}

bool HMD::assignCoarseLocalPermToGlobal(HMD::HMDNode &node,
                                        std::vector<int> &perm) {

  for (int local_node = 0; local_node < node.DOFs.size(); local_node++) {
    int global_node = node.DOFs[local_node];
    perm[node.permuted_new_label[local_node] + node.offset] = global_node;
  }

  if (node.left_node_idx != -1) {
    assignCoarseLocalPermToGlobal(this->HMD_tree[node.left_node_idx], perm);
  }
  if (node.right_node_idx != -1) {
    assignCoarseLocalPermToGlobal(this->HMD_tree[node.right_node_idx], perm);
  }
  return true;
}

// ======================== Tree traversal functions ==========================
bool HMD::isSeparator(int separator_id, int node_id) {
  if (separator_id > node_id) {
    std::cerr << "isSeparator function is not called properly" << std::endl;
    return false;
  }
  int curr_ancestor = (node_id - 1) / 2;
  while (curr_ancestor > separator_id && curr_ancestor > 0) {
    curr_ancestor = (curr_ancestor - 1) / 2;
  }
  if (curr_ancestor == separator_id) {
    return true;
  }
  return false;
}

int HMD::findCommonSeparator(int first, int second) {
  assert(first < num_HMD_nodes);
  assert(second < num_HMD_nodes);
  assert(HMD_tree[first].DOFs.size() != 0);
  assert(HMD_tree[second].DOFs.size() != 0);

  int higher_level_region = first;
  int lower_level_region = second;
  int lower_level = HMD_tree[lower_level_region].level;
  int higher_level = HMD_tree[higher_level_region].level;

  // Make the levels equal
  if (higher_level < lower_level) {
    higher_level_region = second;
    lower_level_region = first;
    lower_level = HMD_tree[lower_level_region].level;
    higher_level = HMD_tree[higher_level_region].level;
  }

  while (HMD_tree[higher_level_region].level !=
         HMD_tree[lower_level_region].level) {
    higher_level_region = HMD_tree[higher_level_region].parent_idx;
    assert(higher_level_region != -1);
  }
  assert(HMD_tree[higher_level_region].level ==
         HMD_tree[lower_level_region].level);

  // Find the common ancestor
  while (HMD_tree[higher_level_region].node_id !=
         HMD_tree[lower_level_region].node_id) {
    higher_level_region = HMD_tree[higher_level_region].parent_idx;
    lower_level_region = HMD_tree[lower_level_region].parent_idx;
    assert(higher_level_region != -1);
    assert(lower_level_region != -1);
  }
  assert(higher_level_region == lower_level_region);
  assert(HMD_tree[higher_level_region].level != -1);
  return higher_level_region;
}

void HMD::unMarkLeftRightSubMeshes(HMDNode &node, std::vector<int> &Marker) {
  if (node.left_node_idx != -1) {
    Marker[node.left_node_idx] = 0;
    auto &left_node = HMD_tree[node.left_node_idx];
    unMarkLeftRightSubMeshes(left_node, Marker);
  }
  if (node.right_node_idx != -1) {
    Marker[node.right_node_idx] = 0;
    auto &right_node = HMD_tree[node.right_node_idx];
    unMarkLeftRightSubMeshes(right_node, Marker);
  }
}

bool HMD::getAssignedDOFsOfCoarseHMD(const HMDNode &HMD_node,
                                     std::vector<int> &assigned_nodes) const {
  assigned_nodes.insert(assigned_nodes.end(), HMD_node.DOFs.begin(),
                        HMD_node.DOFs.end());
  if (HMD_node.left_node_idx != -1) {
    getAssignedDOFsOfCoarseHMD(HMD_tree[HMD_node.left_node_idx],
                               assigned_nodes);
  }
  if (HMD_node.right_node_idx != -1) {
    getAssignedDOFsOfCoarseHMD(HMD_tree[HMD_node.right_node_idx],
                               assigned_nodes);
  }
  return true;
}

void HMD::clearCoarseHMDNode(HMDNode &node) {
  if (node.left_node_idx != -1) {
    auto &left_node = this->HMD_tree[node.left_node_idx];
    clearCoarseHMDNode(left_node);
  }
  if (node.right_node_idx != -1) {
    auto &right_node = this->HMD_tree[node.right_node_idx];
    clearCoarseHMDNode(right_node);
  }
  node.clearNode();
}

///--------------------------------------------------------------------------
/// getCoarseMesh - Given an HMDnNode it will return the submesh of that
/// tree node
///--------------------------------------------------------------------------
void HMD::getCoarseMesh(
    const HMDNode &node, ///<[in] HMD node
    std::vector<int> &
        assigned_nodes, ///<[in] nodes that are assigned to this coarse HMD node
    int M_n,      ///<[in] The size of the mesh that hmd is created based on
    int *Mp,      ///<[in] The pointer array of the mesh
    int *Mi,      ///<[in] the index array of the mesh
    SubMesh &mesh ///<[out] The submesh of the HMD node
) {
  // Find the nodes (do a post order)
  this->getAssignedDOFsOfCoarseHMD(node, assigned_nodes);
  std::vector<int> marked_nodes(M_n, 0);
  for (auto &n : assigned_nodes) {
    marked_nodes[n] = 1;
  }
  this->createSubMesh(marked_nodes, M_n, Mp, Mi, mesh);
}

void HMD::createSubMesh(
    const std::vector<int>
        &chosen_DOFs, ///<[in] a vector of size nodes that defined the su
    const int &M_n,   ///<[in] a vector of size nodes that defined the su
    const int *Mp,    ///<[in] a vector of size nodes that defined the su
    const int *Mi,    ///<[in] a vector of size nodes that defined the su
    SubMesh &sub_mesh) {
  sub_mesh.clear();
  std::vector<int> global_to_local_DOF_id(M_n, -1);
  // Decompose the mesh
  for (int col = 0; col < M_n; col++) {
    if (chosen_DOFs[col] == 1) {
      sub_mesh.Mp.emplace_back(sub_mesh.M_nnz);
      for (int nbr_ptr = Mp[col]; nbr_ptr < Mp[col + 1]; nbr_ptr++) {
        int neighbor = Mi[nbr_ptr];
        if (chosen_DOFs[neighbor] == 1) {
          sub_mesh.Mi.emplace_back(neighbor);
          sub_mesh.M_nnz++;
        }
      }
      sub_mesh.local_to_global_DOF_id.emplace_back(col);
      sub_mesh.M_n++;
    }
  }

  // Final initialization and finish up of the sub mesh creation
  sub_mesh.Mp.emplace_back(sub_mesh.M_nnz);
  // Use to map the mesh nodes' id to local nodes' id
  for (int l_id = 0; l_id < sub_mesh.local_to_global_DOF_id.size(); l_id++) {
    global_to_local_DOF_id[sub_mesh.local_to_global_DOF_id[l_id]] = l_id;
  }
  // Mapping the Mi array from global ids to local ids
  for (auto &row : sub_mesh.Mi) {
    assert(global_to_local_DOF_id[row] != -1);
    row = global_to_local_DOF_id[row];
  }
  assert(sub_mesh.local_to_global_DOF_id.size() == sub_mesh.M_n);
  assert(sub_mesh.Mp.size() == sub_mesh.M_n + 1);
  assert(sub_mesh.Mi.size() == sub_mesh.M_nnz);
  assert(sub_mesh.Mp.back() == sub_mesh.M_nnz);
}

int HMD::getCoarseHMDNodeOffset(const HMDNode &HMD_node) {
  int current_node_idx = HMD_node.node_id;
  while (HMD_tree[current_node_idx].left_node_idx != -1) {
    current_node_idx = HMD_tree[current_node_idx].left_node_idx;
  }

  if (HMD_tree[current_node_idx].right_node_idx != -1) {
    return getCoarseHMDNodeOffset(
        HMD_tree[HMD_tree[current_node_idx].right_node_idx]);
  }
  return HMD_tree[current_node_idx].offset;
}

void HMD::createMultiSubMesh(
    std::vector<int>
        &chosen_region_flag, ///<[in] a number of regions vector where regions
    ///< that should be submeshed is marked as 1
    const std::vector<int> &nodes_regions, ///<[in] a vector of size nodes that
    ///< defined the region of each node
    const int &M_n, /// <[in] Number of nodes inside the full mesh
    const int *Mp,  /// <[in] full mesh pointer array in CSC format
    const int *Mi,  /// <[in] full mesh index array in CSC format
    std::vector<SubMesh>
        &regions_stack /// <[out] full mesh index array in CSC format
) {
#ifndef NDEBUG
  //  if (verbose) {
//    std::cout << "PARTH: createMultiSubMesh: Create submeshes from the
//    regions"
//              << std::endl;
//  }
#endif
  assert(nodes_regions.size() == M_n);
  assert(chosen_region_flag.size() != 0);
  assert(Mp != nullptr);
  assert(Mi != nullptr);
  int num_regions = chosen_region_flag.size();
  regions_stack.resize(num_regions);
  // Clean the chosen submeshes to be recomputed
  for (int i = 0; i < num_regions; i++) {
    if (chosen_region_flag[i] == 1) {
      regions_stack[i].clear();
    }
  }

  // Decompose the mesh
  for (int col = 0; col < M_n; col++) {
    const int &current_region = nodes_regions[col];
    // Is it in chosen regions?
    if (chosen_region_flag[current_region] == 1) {
      // Grab the region
      auto &submesh = regions_stack[current_region];
      submesh.Mp.emplace_back(submesh.M_nnz);
      for (int nbr_ptr = Mp[col]; nbr_ptr < Mp[col + 1]; nbr_ptr++) {
        int neighbor = Mi[nbr_ptr];
        if (nodes_regions[neighbor] == current_region) {
          submesh.Mi.emplace_back(neighbor);
          submesh.M_nnz++;
        }
      }
      submesh.local_to_global_DOF_id.emplace_back(col);
      submesh.M_n++;
    }
  }

  std::vector<int> global_to_local_DOF_id(M_n, -1);
  // Final clean up
  // Final initialization and finish up of the sub mesh creation
  for (int cnt = 0; cnt < num_regions; cnt++) {
    if (chosen_region_flag[cnt] == 1) {
      auto &region = regions_stack[cnt];
      region.Mp.emplace_back(region.M_nnz);
      // Use to map the mesh nodes' id to local nodes' id
      for (int l_id = 0; l_id < region.local_to_global_DOF_id.size(); l_id++) {
        global_to_local_DOF_id[region.local_to_global_DOF_id[l_id]] = l_id;
      }
      // Mapping the Mi array from global ids to local ids
      for (auto &row : region.Mi) {
        assert(global_to_local_DOF_id[row] != -1);
        row = global_to_local_DOF_id[row];
      }
      assert(region.local_to_global_DOF_id.size() == region.M_n);
      assert(region.Mp.size() == region.M_n + 1);
      assert(region.Mi.size() == region.M_nnz);
      assert(region.Mp.back() == region.M_nnz);
    }
  }
}

void HMD::createTriSubMesh(
    const std::vector<int> &nodes_regions, ///<[in] a vector of size nodes that
    ///< defined the region of each node
    const int &M_n,      /// <[in] Number of nodes inside the full mesh
    const int *Mp,       /// <[in] full mesh pointer array in CSC format
    const int *Mi,       /// <[in] full mesh index array in CSC format
    SubMesh &left_mesh,  /// <[in] left submesh
    SubMesh &right_mesh, /// <[in] right_submesh
    SubMesh &sep_mesh    /// <[in] separation mesh
) {
#ifndef NDEBUG
  //  if (verbose) {
//    std::cout << "PARTH: createTriSubMesh: Create submeshes from the 3
//    regions"
//              << std::endl;
//  }
#endif

  assert(nodes_regions.size() == M_n);
  assert(Mp != nullptr);
  assert(Mi != nullptr);

  int num_regions = 3;
  std::vector<SubMesh *> regions_stack{&left_mesh, &right_mesh, &sep_mesh};

  // Clean the chosen submeshes to be recomputed
  for (int i = 0; i < num_regions; i++) {
    regions_stack[i]->clear();
  }

  // Decompose the mesh
  for (int col = 0; col < M_n; col++) {
    const int &current_region = nodes_regions[col];
    // Grab the region
    auto &submesh = *regions_stack[current_region];
    submesh.Mp.emplace_back(submesh.M_nnz);
    for (int nbr_ptr = Mp[col]; nbr_ptr < Mp[col + 1]; nbr_ptr++) {
      int neighbor = Mi[nbr_ptr];
      if (nodes_regions[neighbor] == current_region) {
        submesh.Mi.emplace_back(neighbor);
        submesh.M_nnz++;
      }
    }
    submesh.local_to_global_DOF_id.emplace_back(col);
    submesh.M_n++;
  }

  // Final clean up
  // Final initialization and finish up of the sub mesh creation
  std::vector<int> global_to_local_node_id(M_n);
  for (int cnt = 0; cnt < num_regions; cnt++) {
    auto &region = *regions_stack[cnt];
    region.Mp.emplace_back(region.M_nnz);
    // Use to map the mesh nodes' id to local nodes' id
    for (int l_id = 0; l_id < region.local_to_global_DOF_id.size(); l_id++) {
      global_to_local_node_id[region.local_to_global_DOF_id[l_id]] = l_id;
    }
    // Mapping the Mi array from global ids to local ids
    for (auto &row : region.Mi) {
      row = global_to_local_node_id[row];
    }
    assert(region.local_to_global_DOF_id.size() == region.M_n);
    assert(region.Mp.size() == region.M_n + 1);
    assert(region.Mi.size() == region.M_nnz);
    assert(region.Mp.back() == region.M_nnz);
  }
}
void HMD::updateFullOffsetRecursive(double &accumulated_offset, HMDNode &node) {
  // Do a DFS strating from root to the left most node
  if (node.left_node_idx != -1) {
    updateFullOffsetRecursive(accumulated_offset, HMD_tree[node.left_node_idx]);
  }
  if (node.right_node_idx != -1) {
    updateFullOffsetRecursive(accumulated_offset,
                              HMD_tree[node.right_node_idx]);
  }
  node.offset = accumulated_offset;
  accumulated_offset += node.DOFs.size();
}

void HMD::updateCoarseHMDNodeOffset(int node_idx) {
  double offset = getCoarseHMDNodeOffset(HMD_tree[node_idx]);
  updateFullOffsetRecursive(offset, HMD_tree[0]);
}

int HMD::getTotalEdgesInHMDNode(int node_idx, int M_n, int *Mp) {
  std::vector<int> &DOFs = HMD_tree[node_idx].DOFs;
  int total_edges = 0;
  for (auto &node : DOFs) {
    assert(node < M_n);
    total_edges += Mp[node + 1] - Mp[node];
  }
  return total_edges;
}

/// getCoarseNodeDOFsSize - Get the size of the DOFs assigned to the coarse HMD
/// node
int HMD::getCoarseNodeDOFsSize(int hmd_node_idx) {
  if (hmd_node_idx >= num_HMD_nodes) {
    std::cerr << "The node index is out of range" << std::endl;
    return 0;
  }
  int total_num_dofs = 0;
  std::queue<int> node_queue;
  node_queue.push(hmd_node_idx);
  // Doing a BFS on HMD_tree
  while (!node_queue.empty()) {
    int current_node_idx = node_queue.front();
    node_queue.pop();
    if (current_node_idx >= num_HMD_nodes) {
      continue;
    }
    auto &current_node = HMD_tree[current_node_idx];
    total_num_dofs += current_node.DOFs.size();
    if (current_node.left_node_idx != -1) {
      node_queue.push(current_node.left_node_idx);
    }
    if (current_node.right_node_idx != -1) {
      node_queue.push(current_node.right_node_idx);
    }
  }
  return total_num_dofs;
}

//// Computing Parallel Schedule with SpMP for decomposing
//void HMD::computeP2PSchedule(int start_node, int num_HMD_nodes) {
//  int map_idx = num_HMD_nodes - 1;
//  assert(HMD_tree.size() != 0);
//  // Create the dependency graph of HMD binary tree
//  // Compute the pairs of entries
//  std::vector<std::pair<int, int>> pairs;
//  std::queue<int> bfs_queue;
//  bfs_queue.push(start_node);
//  while (!bfs_queue.empty()) {
//    int current_node_idx = bfs_queue.front();
//    int left_node_idx = current_node_idx * 2 + 1;
//    int right_node_idx = current_node_idx * 2 + 2;
//    pairs.emplace_back(map_idx - current_node_idx,
//                       map_idx - current_node_idx);
//    if (left_node_idx < num_HMD_nodes) {
//      pairs.emplace_back(map_idx - current_node_idx,
//                         map_idx - left_node_idx);
//      pairs.emplace_back(map_idx - left_node_idx,
//                         map_idx - current_node_idx);
//      bfs_queue.push(left_node_idx);
//    }
//    if (right_node_idx < num_HMD_nodes) {
//      pairs.emplace_back(map_idx - current_node_idx,
//                         map_idx - right_node_idx);
//      pairs.emplace_back(map_idx - right_node_idx,
//                         map_idx - current_node_idx);
//      bfs_queue.push(right_node_idx);
//    }
//    bfs_queue.pop();
//  }
//
//  // Create the sparse matrix
//  std::vector<int> tree_ptr;
//  std::vector<int> tree_idx;
//  coo_to_csr(pairs, num_HMD_nodes, tree_ptr, tree_idx);
//
//  auto *HMD_binary_tree = new SpMP::CSR();
//  HMD_binary_tree->alloc(num_HMD_nodes, tree_ptr[num_HMD_nodes], true);
//  HMD_binary_tree->n = num_HMD_nodes;
//  std::copy(tree_ptr.begin(), tree_ptr.end(), HMD_binary_tree->rowptr);
//  std::copy(tree_idx.begin(), tree_idx.end(), HMD_binary_tree->colidx);
//  std::fill(HMD_binary_tree->values,
//            HMD_binary_tree->values + tree_ptr[num_HMD_nodes], 1);
//
//  int base = 0;
//#pragma omp parallel for
//  for (int i = 0; i < tree_ptr.size() - 1; ++i) {
//    for (int j = HMD_binary_tree->rowptr[i] - base;
//         j < HMD_binary_tree->rowptr[i + 1] - base; ++j) {
//      if (HMD_binary_tree->colidx[j] - base == i) {
//        HMD_binary_tree->diagptr[i] = j + base;
//        HMD_binary_tree->idiag[i] = 1 / HMD_binary_tree->values[j];
//        HMD_binary_tree->diag[i] = HMD_binary_tree->values[j];
//      }
//    }
//  }
//
//#ifndef NDEBUG
//  int *symRowPtr = NULL, *symColIdx = NULL, *symDiagPtr = NULL,
//      *symExtPtr = NULL;
//  assert(getSymmetricNnzPattern(HMD_binary_tree, &symRowPtr, &symDiagPtr,
//                                &symExtPtr, &symColIdx));
//  FREE(symRowPtr);
//  FREE(symColIdx);
//  FREE(symDiagPtr);
//  FREE(symExtPtr);
//#endif
//
//  // Create the schedule
//  p2pScheduleForDecomposing = new SpMP::LevelSchedule;
//  p2pScheduleForDecomposing->transitiveReduction = false;
//  p2pScheduleForDecomposing->constructTaskGraph(*HMD_binary_tree);
//  assert(p2pScheduleForDecomposing->origToThreadContPerm != nullptr);
//  assert(p2pScheduleForDecomposing->threadContToOrigPerm != nullptr);
//
//  delete HMD_binary_tree;
//}



int HMD::computeHMDLVL(int node_idx) {
  int lvl = 0;
  while (node_idx > 0) {
    node_idx = (node_idx - 1) / 2;
    lvl++;
  }
  return lvl;
}

} // namespace PARTH
