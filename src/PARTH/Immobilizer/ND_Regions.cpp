//
// Created by behrooz on 10/11/22.
//

#include "ND_Regions.h"
#include <algorithm>
#include <math.h>
#include <set>
#include <tuple>
#include <omp.h>

namespace PARTH {

void PARTH::ND_Regions::initRegions(
    int num_levels, /// <[in] How many levels you want to dissect the graph
    int M_n,        ///<[in] Number of elements inside the mesh
    int *Mp,        ///<[in] Pointer array of mesh in CSC format
    int *Mi,        ///<[in] Index array of mesh in CSC format
    std::vector<int> &mesh_perm ///<[out] The mesh permutation array
) {

  assert(num_levels >= 0);
  if (num_levels > 12) {
    num_levels = 10;
  }
  this->num_levels = num_levels;
  this->M_n = M_n;
  this->Mp = Mp;
  this->Mi = Mi;
  this->num_tree_nodes = std::pow(2, num_levels + 1) - 1;
  this->tree_nodes.resize(num_tree_nodes);
  this->global_to_local_node_id.resize(M_n);
  this->nodes_regions.resize(M_n, 0);
  std::vector<int> assigned_nodes(M_n);
  for (int i = 0; i < M_n; i++) {
    assigned_nodes[i] = i;
  }

  this->parallel_analysis = true;
  this->ND_Grouping(0, -1, 0, M_n, Mp, Mi, assigned_nodes, 0);
  mesh_perm.resize(M_n);

  assignSubtreePermToGlobal(tree_nodes[0], mesh_perm);
  // Find the balance parameter
  //    getSepQualityPar(tree_nodes[0], sep_par, child_par,
  //    sep_balance_node_id);
  cached_regions_flag.resize(num_tree_nodes, 0);
  this->regions_init = true;
}

bool PARTH::ND_Regions::ND_Grouping_recursive(
    int tree_node_id,   ///[in] node id
    int parent_node_id, /// <[in] parent node id
    int current_level,  ///<[in] The current level that we are dissecting
    int M_n,            ///<[in] Number of nodes in the mesh
    int *Mp,            ///<[in] The pointer array of mesh
    int *Mi,            ///<[in] The index array of mesh
    std::vector<int> &assigned_nodes, ///<[in] assigned node to this tree node
    int offset ///<[in] number of nodes before this region

) {

  auto &current_node = this->tree_nodes[tree_node_id];
  //+++++++++++++ Boundary condition of a single level ++++++++++++++++
  if (current_level == num_levels) {
    current_node.setNodeIdentifiers(-1, -1, tree_node_id, parent_node_id,
                                    offset, current_level);
    // Init the assigned nodes
    current_node.assigned_nodes = assigned_nodes;
    current_node.setAssignedNodesInitFlag();

    // Assign the region to the global region
    for (auto &m_node : assigned_nodes) {
      this->nodes_regions[m_node] = tree_node_id;
    }

    // Assign the permuted labels to the sep nodes
    std::vector<int> perm(M_n);
    metisNodeNDPermutation(M_n, Mp, Mi, perm.data(), nullptr);
    current_node.setPermutedNewLabel(perm);

    assert(current_node.treeNodeIsDefined());
    return true;
  }

  std::vector<int> local_nodes_regions(M_n);
  idx_t nVertices = M_n;
  idx_t csp;
  std::vector<int> vweight(nVertices, 1);
  int ret =
      METIS_ComputeVertexSeparator(&nVertices, Mp, Mi, vweight.data(), NULL,
                                   &csp, local_nodes_regions.data());

  if (ret != METIS_OK) {
    std::cerr << "Something went wrong" << std::endl;
    return false;
  }

  // More level is needed
  std::vector<SubMesh> regions_stack;
  std::vector<int> chosen_regions_flags{1, 1, 1};
  createMultiSubMesh(chosen_regions_flags, local_nodes_regions, M_n, Mp, Mi,
                     regions_stack);

  auto &left_region = regions_stack[0];
  auto &right_region = regions_stack[1];
  auto &sep_region = regions_stack[2];

  // Assign nodes
  std::vector<int> left_assigned;
  std::vector<int> right_assigned;
  auto &sep_assigned = current_node.assigned_nodes;
  for (int i = 0; i < local_nodes_regions.size(); i++) {
    if (local_nodes_regions[i] == 0) { // Left assigned
      left_assigned.emplace_back(assigned_nodes[i]);
    } else if (local_nodes_regions[i] == 1) {
      right_assigned.emplace_back(assigned_nodes[i]);
    } else if (local_nodes_regions[i] == 2) {
      sep_assigned.emplace_back(assigned_nodes[i]);
    } else {
      std::cerr << "There are more than 3 regions in here" << std::endl;
      return false;
    }
  }

  // Assign the permuted labels to the sep nodes
  sep_region.perm.resize(sep_region.M_n);
  metisNodeNDPermutation(sep_region.M_n, sep_region.Mp.data(),
                         sep_region.Mi.data(), sep_region.perm.data(), nullptr);

  current_node.setNodeIdentifiers(
      tree_node_id * 2 + 1, tree_node_id * 2 + 2, tree_node_id, parent_node_id,
      offset + left_region.M_n + right_region.M_n, current_level);
  current_node.setAssignedNodesInitFlag();
  current_node.setPermutedNewLabel(sep_region.perm);

  // assign the regions to the global node regions
  for (auto &sep_node : sep_assigned) {
    this->nodes_regions[sep_node] = tree_node_id;
  }

  if (left_region.M_n != 0) {
    ND_Grouping_recursive(current_node.left_node_idx, current_node.node_id,
                          current_level + 1, left_region.M_n,
                          left_region.Mp.data(), left_region.Mi.data(),
                          left_assigned, offset);
  }

  if (right_region.M_n != 0) {
    ND_Grouping_recursive(current_node.right_node_idx, current_node.node_id,
                          current_level + 1, right_region.M_n,
                          right_region.Mp.data(), right_region.Mi.data(),
                          right_assigned, offset + left_region.M_n);
  }
  return true;
}

bool PARTH::ND_Regions::ND_Grouping_parallel(
    int node_id,   ///[in] node id
    int p_id,      /// <[in] parent node id
    int cur_level, ///<[in] The current level that we are dissecting
    int M_n,       ///<[in] Number of nodes in the mesh
    int *Mp,       ///<[in] The pointer array of mesh
    int *Mi,       ///<[in] The index array of mesh
    std::vector<int>
        &sub_tree_assigned, ///<[in] assigned node to this tree node
    int offset              ///<[in] number of nodes before this region
) {
  assert(M_n != 0);
  // Boundary, if first node, init the first node
  auto &cur_node = this->tree_nodes[node_id];
  //+++++++++++++ Boundary condition of a single level ++++++++++++++++
  if (cur_level == num_levels) {
    cur_node.setNodeIdentifiers(-1, -1, node_id, p_id, offset, cur_level);
    // Init the assigned nodes
    cur_node.assigned_nodes = sub_tree_assigned;
    cur_node.setAssignedNodesInitFlag();

    // Assign the region to the global region
    for (auto &m_node : sub_tree_assigned) {
      this->nodes_regions[m_node] = node_id;
    }

    // Assign the permuted labels to the sep nodes
    std::vector<int> perm(M_n);
    metisNodeNDPermutation(M_n, Mp, Mi, perm.data(), nullptr);
    cur_node.setPermutedNewLabel(perm);

    assert(cur_node.treeNodeIsDefined());
    return true;
  }

  // Prepare the Wavefront parallelism arrays and data
  int total_number_of_sub_tree_nodes =
      std::pow(2, (num_levels - cur_level) + 1) - 1;
  // Levels are zero based. So for num_levels = 1 we have two levels 0 and 1,
  // and we need
  //  an array of size 3 to define wavefront parallelism
  int wavefront_levels = num_levels - cur_level + 1;

  std::vector<int> level_ptr(wavefront_levels + 1);
  std::vector<int> tree_node_ids_set(total_number_of_sub_tree_nodes);
  std::vector<int> tree_node_ids_set_inv(this->num_tree_nodes);
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

  tree_node_ids_set[0] = node_id;
  parent_node_ids_set[0] = p_id;
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
  sub_mesh_assigned_nodes[0] = sub_tree_assigned;
  offset_set[0] = offset;

  // start the wavefront parallelism
#pragma omp parallel
  {
    for (int l = 0; l < wavefront_levels; l++) {
#pragma omp for
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
        auto &current_node = this->tree_nodes[id];
        //+++++++++++++ Boundary condition of a single level ++++++++++++++++
        if (current_level == num_levels || mesh.M_nnz == 0) {
          current_node.setNodeIdentifiers(-1, -1, id, parent_id, offset_par,
                                          current_level);
          // Init the assigned nodes
          current_node.assigned_nodes = assigned_nodes_par;
          current_node.setAssignedNodesInitFlag();

          // Assign the region to the global region
          for (auto &m_node : assigned_nodes_par) {
            this->nodes_regions[m_node] = id;
          }

          // Assign the permuted labels to the sep nodes
          std::vector<int> perm(M_n);
          metisNodeNDPermutation(M_n, Mp, Mi, perm.data(), nullptr);
          current_node.setPermutedNewLabel(perm);

          assert(current_node.treeNodeIsDefined());
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
        auto &sep_assigned = current_node.assigned_nodes;
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
        metisNodeNDPermutation(sep_region.M_n, sep_region.Mp.data(),
                               sep_region.Mi.data(), sep_region.perm.data(),
                               nullptr);

        current_node.setNodeIdentifiers(
            id * 2 + 1, id * 2 + 2, id, parent_id,
            offset_par + left_region.M_n + right_region.M_n, current_level);
        current_node.setAssignedNodesInitFlag();
        current_node.setPermutedNewLabel(sep_region.perm);

        // assign the regions to the global node regions
        for (auto &sep_node : sep_assigned) {
          this->nodes_regions[sep_node] = id;
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

void PARTH::ND_Regions::createMultiSubMesh(
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

  // Final clean up
  // Final initialization and finish up of the sub mesh creation
  for (int cnt = 0; cnt < num_regions; cnt++) {
    if (chosen_region_flag[cnt] == 1) {
      auto &region = regions_stack[cnt];
      region.Mp.emplace_back(region.M_nnz);
      // Use to map the mesh nodes' id to local nodes' id
      for (int l_id = 0; l_id < region.local_to_global_DOF_id.size(); l_id++) {
        global_to_local_node_id[region.local_to_global_DOF_id[l_id]] = l_id;
      }
      // Mapping the Mi array from global ids to local ids
      for (auto &row : region.Mi) {
        row = global_to_local_node_id[row];
      }
      assert(region.local_to_global_node_id.size() == region.M_n);
      assert(region.Mp.size() == region.M_n + 1);
      assert(region.Mi.size() == region.M_nnz);
      assert(region.Mp.back() == region.M_nnz);
    }
  }
}

void PARTH::ND_Regions::createTriSubMesh(
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
    assert(region.local_to_global_node_id.size() == region.M_n);
    assert(region.Mp.size() == region.M_n + 1);
    assert(region.Mi.size() == region.M_nnz);
    assert(region.Mp.back() == region.M_nnz);
  }
}

void ND_Regions::getGlobalPerm(std::vector<int> &contact_points,
                               std::vector<int> &mesh_perm) {
  if (mesh_perm.empty()) {
    std::cerr << "PARTH: getGlobalPerm - mesh_perm should be allocated first"
              << std::endl;
  }

#ifndef NDEBUG
  for (int i = 0; i < M_n; i++) {
    int current_region = nodes_regions[i];
    for (int nbr_ptr = Mp[i]; nbr_ptr < Mp[i + 1]; nbr_ptr++) {
      int nbr = Mi[nbr_ptr];
      int nbr_region = nodes_regions[nbr];
      int big_id_region = current_region;
      int small_id_region = nbr_region;
      if (small_id_region > big_id_region) {
        int tmp = big_id_region;
        big_id_region = small_id_region;
        small_id_region = tmp;
      }
      if (small_id_region == big_id_region) {
        continue;
      } else if (isAncestor(small_id_region, big_id_region)) {
        continue;
      } else {
        if (std::find(contact_points.begin(), contact_points.end(), i) ==
            contact_points.end()) {
          assert(false);
        }
        if (std::find(contact_points.begin(), contact_points.end(), nbr) ==
            contact_points.end()) {
          assert(false);
        }
      }
    }
  }
#endif
  this->dirty_tree_node_idx.clear();
  this->dirty_node_idx.clear();
  this->cached_regions_flag.clear();
  cached_regions_flag.resize(this->num_tree_nodes, 1);
  if (contact_points.empty()) {
    return;
  }
  findDirtyTreeNodes_time = omp_get_wtime();
  std::set<int> dirty_region_idx;
  std::set<std::tuple<int, int>> edge_set;
  findRegionConnections(contact_points, num_contact_points, dirty_node_idx,
                        edge_set);
  std::vector<int> edge_Set_flag;
  std::vector<int> common_ancestor_flag;
  std::vector<std::tuple<int, int>> unresolved_edge_set;
  for (auto &edge : edge_set) {
    //    if (!moveContactEdgeToCommonSeparator(std::get<0>(edge),
    //    std::get<1>(edge),
    //                                          common_ancestor_flag)) {
    unresolved_edge_set.emplace_back(edge);
    //    }
    //    if (common_ancestor_flag[0] == 1) {
    //      break;
    //    }
  }
  findDirtyTreeNodes_time = omp_get_wtime() - findDirtyTreeNodes_time;
  filterRegionConnectios(unresolved_edge_set);
  // If the root node of the partition tree is still dirty
  if (getReuseRatio() < 0.2) {
    dirty_node_idx.clear();
    dirty_tree_node_idx.clear();
    dirty_tree_node_idx.emplace_back(0);
    auto &tree_node = this->tree_nodes[0];
    fixSubtreePartition(tree_node);
    // Assemble the permutation
    assignSubtreePermToGlobal(tree_node, mesh_perm);
  } else {
    /*Filter the unresolved_edge_set and dirty_node_idx
     (that is if a local change in the region also is part of the subtree
     that should be repartitioned, then delete that from dirty_node_idx
     */
    // These are fix in permutation per region
    fixAndAssignRegionPermToGlobalPerm(dirty_node_idx, mesh_perm);

    // TODO: Parallelize this loop later
    // These are fix in permutations per subtree
    dirty_tree_node_idx_saved = dirty_tree_node_idx;
    for (auto &tree_node_idx : dirty_tree_node_idx) {
      auto &tree_node = this->tree_nodes[tree_node_idx];
      fixSubtreePartition(tree_node);
      // Assemble the permutation
      assignSubtreePermToGlobal(tree_node, mesh_perm);
    }
  }

#ifndef NDEBUG
  for (int i = 0; i < M_n; i++) {
    int current_region = nodes_regions[i];
    for (int nbr_ptr = Mp[i]; nbr_ptr < Mp[i + 1]; nbr_ptr++) {
      int nbr = Mi[nbr_ptr];
      int nbr_region = nodes_regions[nbr];
      int big_id_region = current_region;
      int small_id_region = nbr_region;
      if (small_id_region > big_id_region) {
        int tmp = big_id_region;
        big_id_region = small_id_region;
        small_id_region = tmp;
      }
      if (small_id_region == big_id_region) {
        continue;
      } else if (isAncestor(small_id_region, big_id_region)) {
        continue;
      } else {
        assert(false);
      }
    }
  }
#endif
}

///--------------------------------------------------------------------------
/// fixRegionPartition - recompute the permutation of the region represented
/// by the node
///--------------------------------------------------------------------------
void ND_Regions::fixAndAssignRegionPermToGlobalPerm(
    std::vector<int> &dirty_regions_idx, std::vector<int> &mesh_perm) {

  if (dirty_regions_idx.empty()) {
    return;
  }
  assert(!mesh_perm.empty());

  std::vector<int> region_marker(this->num_tree_nodes, 0);
  for (auto &region : dirty_regions_idx) {
    region_marker[region] = 1;
  }
  std::vector<SubMesh> region_stack;

  createMultiSubMesh(region_marker, this->nodes_regions, M_n, Mp, Mi,
                     region_stack);

#pragma omp parallel for
  for (int i = 0; i < dirty_regions_idx.size(); i++) {
    // Fix the permutation
    int region_idx = dirty_regions_idx[i];
    auto &node = tree_nodes[region_idx];
    auto &region_mesh = region_stack[region_idx];
    cached_regions_flag[node.node_id] = 0;
    // Assign the permuted labels to the sep nodes
    std::vector<int> perm(region_mesh.M_n);
    metisNodeNDPermutation(region_mesh.M_n, region_mesh.Mp.data(),
                           region_mesh.Mi.data(), perm.data(), nullptr);
    node.setPermutedNewLabel(perm);

    // Assign the permutation to the global node
    for (int local_node = 0; local_node < node.assigned_nodes.size();
         local_node++) {
      int global_node = node.assigned_nodes[local_node];
      mesh_perm[node.permuted_new_label[local_node]] = global_node;
    }
  }
}

void filterRegionConnectios(
    std::vector<std::tuple<int, int>>
        &edge_set, ///<[out] an edge that shows the connection ///< between
                   ///< two regions
    std::vector<int>
        &within_region_connections ///<[out] the region idx that has within
                                   ///< region connections
) {
  std::cerr << "Implement filterRegionConnectios" << std::endl;
}

void ND_Regions::fixSubtreePartition(NestedDissectionNode &node) {
  SubMesh local_mesh;
  std::vector<int> assigned_nodes;
  double tmp_timer = 0;
  if (node.node_id == 0) {
    clearSubTree(node);
    std::vector<int> assigned_nodes(M_n);
    for (int i = 0; i < M_n; i++) {
      assigned_nodes[i] = i;
    }
    tmp_timer = omp_get_wtime();
    ND_Grouping(0, -1, 0, M_n, Mp, Mi, assigned_nodes, 0);
    // Find the balance parameter
    getSepQualityPar(tree_nodes[0], sep_par, child_par, sep_balance_node_id);
    tmp_timer = omp_get_wtime() - tmp_timer;
    ND_Grouping_time += tmp_timer;
    cached_regions_flag.resize(this->num_tree_nodes);
    std::fill(cached_regions_flag.begin(), cached_regions_flag.end(), 0);
    return;
  }

  tmp_timer = omp_get_wtime();
  getTreeNodeMesh(node, local_mesh, assigned_nodes);
  tmp_timer = omp_get_wtime() - tmp_timer;
  getTreeNodeMesh_time += tmp_timer;

  tmp_timer = omp_get_wtime();
  std::sort(assigned_nodes.begin(), assigned_nodes.end());
  tmp_timer = omp_get_wtime() - tmp_timer;
  sort_assign_time += tmp_timer;

  tmp_timer = omp_get_wtime();
  int start_offset = getSubTreeOffset(node);
  tmp_timer = omp_get_wtime() - tmp_timer;
  getSubTreeOffset_time += tmp_timer;

  assert(start_offset >= 0 && start_offset < M_n);
  // clearSubTree should be called after start_offset
  int node_id = node.node_id;
  int parent_idx = node.parent_idx;
  int level = node.level;

  tmp_timer = omp_get_wtime();
  clearSubTree(node);
  unMarkDescendent(node, cached_regions_flag);
  tmp_timer = omp_get_wtime() - tmp_timer;
  clearSubTree_time += tmp_timer;

#ifndef NDEBUG
  std::cout << "PARTH: fixLocalPartition - fixing the partition location"
            << std::endl;
#endif

  tmp_timer = omp_get_wtime();
  ND_Grouping(node_id, parent_idx, level, local_mesh.M_n, local_mesh.Mp.data(),
              local_mesh.Mi.data(), assigned_nodes, start_offset);
  tmp_timer = omp_get_wtime() - tmp_timer;
  ND_Grouping_time += tmp_timer;
}

///--------------------------------------------------------------------------
/// getTreeNodeMesh - Given a tree node it will return the submesh of that
/// tree node
///--------------------------------------------------------------------------
void ND_Regions::getTreeNodeMesh(const NestedDissectionNode &node,
                                 SubMesh &mesh,
                                 std::vector<int> &assigned_nodes) {
  // Find the nodes (do a post order)
  this->getAssignedNodeofSubtree(node, assigned_nodes);
  std::vector<int> marked_nodes(M_n, 0);
  for (auto &n : assigned_nodes) {
    marked_nodes[n] = 1;
  }
  this->createSubMesh(marked_nodes, M_n, Mp, Mi, mesh);
}

void ND_Regions::createSubMesh(
    const std::vector<int>
        &chosen_nodes, ///<[in] a vector of size nodes that defined the su
    const int &M_n,    ///<[in] a vector of size nodes that defined the su
    const int *Mp,     ///<[in] a vector of size nodes that defined the su
    const int *Mi,     ///<[in] a vector of size nodes that defined the su
    SubMesh &sub_mesh) {
  sub_mesh.clear();
  // Decompose the mesh
  for (int col = 0; col < M_n; col++) {
    if (chosen_nodes[col] == 1) {
      sub_mesh.Mp.emplace_back(sub_mesh.M_nnz);
      for (int nbr_ptr = Mp[col]; nbr_ptr < Mp[col + 1]; nbr_ptr++) {
        int neighbor = Mi[nbr_ptr];
        if (chosen_nodes[neighbor] == 1) {
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
    this->global_to_local_node_id[sub_mesh.local_to_global_DOF_id[l_id]] =
        l_id;
  }
  // Mapping the Mi array from global ids to local ids
  for (auto &row : sub_mesh.Mi) {
    row = global_to_local_node_id[row];
  }
  assert(sub_mesh.local_to_global_node_id.size() == sub_mesh.M_n);
  assert(sub_mesh.Mp.size() == sub_mesh.M_n + 1);
  assert(sub_mesh.Mi.size() == sub_mesh.M_nnz);
  assert(sub_mesh.Mp.back() == sub_mesh.M_nnz);
}

bool ND_Regions::getAssignedNodeofSubtree(
    const NestedDissectionNode &node, std::vector<int> &assigned_nodes) const {
  assigned_nodes.insert(assigned_nodes.end(), node.assigned_nodes.begin(),
                        node.assigned_nodes.end());
  if (node.left_node_idx != -1) {
    getAssignedNodeofSubtree(this->tree_nodes[node.left_node_idx],
                             assigned_nodes);
  }
  if (node.right_node_idx != -1) {
    getAssignedNodeofSubtree(this->tree_nodes[node.right_node_idx],
                             assigned_nodes);
  }
  return true;
}

void ND_Regions::metisNodeNDPermutation(int M_n, int *Mp, int *Mi, int *perm,
                                        int *Iperm) {
  idx_t N = M_n;
  idx_t NNZ = Mp[M_n];
  if (NNZ == 0) {
    for (int i = 0; i < M_n; i++) {
#ifndef NDEBUG
      std::cout << "WARNING: SOME OF THE REGIONS ARE CONSIST OF EMPTY NODES"
                << std::endl;
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

bool ND_Regions::assignSubtreePermToGlobal(
    PARTH::ND_Regions::NestedDissectionNode &node, std::vector<int> &perm) {

  for (int local_node = 0; local_node < node.assigned_nodes.size();
       local_node++) {
    int global_node = node.assigned_nodes[local_node];
    perm[node.permuted_new_label[local_node]] = global_node;
  }

  if (node.left_node_idx != -1) {
    assignSubtreePermToGlobal(this->tree_nodes[node.left_node_idx], perm);
  }
  if (node.right_node_idx != -1) {
    assignSubtreePermToGlobal(this->tree_nodes[node.right_node_idx], perm);
  }
  return true;
}

void ND_Regions::assignMeshPtr(int M_n, int *Mp, int *Mi) {
  this->M_n = M_n;
  this->Mp = Mp;
  this->Mi = Mi;
}

int ND_Regions::getSubTreeOffset(const NestedDissectionNode &node) {
  int current_node_idx = node.node_id;
  while (this->tree_nodes[current_node_idx].left_node_idx != -1) {
    current_node_idx = this->tree_nodes[current_node_idx].left_node_idx;
  }

  if (this->tree_nodes[current_node_idx].right_node_idx != -1) {
    return getSubTreeOffset(this->tree_nodes[current_node_idx]);
  }
  return this->tree_nodes[current_node_idx].offset;
}

void ND_Regions::clearSubTree(NestedDissectionNode &node) {
  if (node.left_node_idx != -1) {
    auto &left_node = this->tree_nodes[node.left_node_idx];
    clearSubTree(left_node);
  }
  if (node.right_node_idx != -1) {
    auto &right_node = this->tree_nodes[node.right_node_idx];
    clearSubTree(right_node);
  }
  node.clearTreeNode();
}

void ND_Regions::findRegionConnections(
    std::vector<int>
        &contact_points,     ///<[in] Current points with changed neighbors
    int &num_contact_points, ///<[out] number of contact points
    std::vector<int>
        &dirty_node_idx, ///<[out] regions that has within region contact points
    std::set<std::tuple<int, int>>
        &edge_set ///<[out] an edge that shows the connection ///< between two
                  ///< regions
) {
  num_contact_points = contact_points.size();

  if (num_contact_points == 0) {
    return;
  }
  if (contact_points.size() != 0) {
    assert(true);
  }
  // Find the connection edges between regions
  std::set<int> within_region_connections;
  for (auto &contact_point : contact_points) {
    int &first_region_id = this->nodes_regions[contact_point];
    for (int nbr_ptr = Mp[contact_point]; nbr_ptr < Mp[contact_point + 1];
         nbr_ptr++) {
      int &second_region_id = this->nodes_regions[Mi[nbr_ptr]];
      int big_id_region = first_region_id;
      int small_id_region = second_region_id;
      if (small_id_region > big_id_region) {
        int tmp = big_id_region;
        big_id_region = small_id_region;
        small_id_region = tmp;
      }
      // We only add neighbors that are not in the region tree
      if (small_id_region == big_id_region) {
        within_region_connections.insert(small_id_region);
        continue;
      } else if (isAncestor(small_id_region, big_id_region)) {
        continue;
      } else {
        edge_set.insert(std::tuple<int, int>(small_id_region, big_id_region));
      }
    }
  }

  // Copying the within region connection into dirty node idx
  for (auto &iter : within_region_connections) {
    dirty_node_idx.emplace_back(iter);
  }
}

void ND_Regions::filterRegionConnectios(
    std::vector<std::tuple<int, int>>
        &edge_set ///<[in] an edge that shows the connection ///< between
                  ///< two regions
) {
  // given each edge, find the common ancestor
  for (auto &edge : edge_set) {
    int first_region = std::get<0>(edge);
    int second_region = std::get<1>(edge);
    this->dirty_tree_node_idx.emplace_back(
        findCommonAncestor(first_region, second_region));
  }

  // Filter the dirty tree nodes based on their ancestor
  std::sort(dirty_tree_node_idx.begin(), dirty_tree_node_idx.end());
  dirty_tree_node_idx.erase(
      std::unique(dirty_tree_node_idx.begin(), dirty_tree_node_idx.end()),
      dirty_tree_node_idx.end());

  // Clean dirty regions of descendents based on their ancestor
  std::vector<int> dirty_tree_nodes(this->num_tree_nodes, 0);
  std::vector<int> dirty_regions(this->num_tree_nodes, 0);
  for (auto &iter : dirty_tree_node_idx) {
    dirty_tree_nodes[iter] = 1;
    dirty_regions[iter] = 1;
  }
  for (auto &iter : dirty_node_idx) {
    dirty_regions[iter] = 1;
  }

  for (auto &node_idx : dirty_tree_node_idx) {
    auto &dirty_node = this->tree_nodes[node_idx];
    if (dirty_tree_nodes[node_idx] == 1) {
      unMarkDescendent(dirty_node, dirty_tree_nodes);
      unMarkDescendent(dirty_node, dirty_regions);
    }
  }

  // clean and filter the dirty_tree_node_idx
  dirty_tree_node_idx.clear();
  for (int i = 0; i < num_tree_nodes; i++) {
    if (dirty_tree_nodes[i] == 1) {
      dirty_tree_node_idx.emplace_back(i);
    }
  }

  // clean and filter the dirty_node_idx
  std::vector<int> dirty_node_idx_tmp = dirty_node_idx;
  dirty_node_idx.clear();
  for (auto &iter : dirty_node_idx_tmp) {
    if (dirty_regions[iter] == 1) {
      dirty_node_idx.emplace_back(iter);
    }
  }
}

void ND_Regions::unMarkDescendent(NestedDissectionNode &node,
                                  std::vector<int> &Marker) {
  if (node.left_node_idx != -1) {
    Marker[node.left_node_idx] = 0;
    auto &left_node = this->tree_nodes[node.left_node_idx];
    unMarkDescendent(left_node, Marker);
  }
  if (node.right_node_idx != -1) {
    Marker[node.right_node_idx] = 0;
    auto &right_node = this->tree_nodes[node.right_node_idx];
    unMarkDescendent(right_node, Marker);
  }
}

double ND_Regions::getReuseRatio() const {
  double nodes = 0;
  for (auto &dirty : dirty_tree_node_idx) {
    std::vector<int> assigned_nodes;
    getAssignedNodeofSubtree(tree_nodes[dirty], assigned_nodes);
    nodes += assigned_nodes.size();
  }

  for (auto &dirty : dirty_node_idx) {
    nodes += tree_nodes[dirty].assigned_nodes.size();
  }

  if (nodes_regions.size() != 0) {
    return 1 - (nodes * 1.0 / nodes_regions.size());
  } else {
    return 0.0;
  }
}

std::vector<int> ND_Regions::getDirtyNodes() {
    std::vector<int> assigned_nodes;
    for (auto &dirty : dirty_tree_node_idx) {
        getAssignedNodeofSubtree(tree_nodes[dirty], assigned_nodes);
    }

    for (auto &dirty : dirty_node_idx) {
        assigned_nodes.insert(assigned_nodes.begin(),
                              tree_nodes[dirty].assigned_nodes.begin(),
                              tree_nodes[dirty].assigned_nodes.end());
    }
    return assigned_nodes;
}

bool ND_Regions::getSubTreeRegions(
        const NestedDissectionNode &node, std::vector<int> &sub_tree_regions)  {
    sub_tree_regions.emplace_back(node.node_id);
    if (node.left_node_idx != -1) {
        getSubTreeRegions(this->tree_nodes[node.left_node_idx],
                                 sub_tree_regions);
    }
    if (node.right_node_idx != -1) {
        getSubTreeRegions(this->tree_nodes[node.right_node_idx],
                                 sub_tree_regions);
    }
    return true;
}

std::vector<int> ND_Regions::getDirtyRegions() {
    double nodes = 0;
    std::vector<int> dirty_regions;
    for (auto &dirty : dirty_tree_node_idx) {
        getSubTreeRegions(tree_nodes[dirty], dirty_regions);
    }

    for (auto &dirty : dirty_node_idx) {
        dirty_regions.emplace_back(dirty);
    }

    return dirty_regions;
}

void ND_Regions::exportAnalysisStatistics(int frame, int iter,
                                          std::string address) {
  double total_time = ND_Grouping_time + clearSubTree_time +
                      getSubTreeOffset_time + sort_assign_time +
                      getTreeNodeMesh_time + findDirtyTreeNodes_time;
  std::vector<std::string> Runtime_headers;
  Runtime_headers.emplace_back("frame");
  Runtime_headers.emplace_back("iter");
  Runtime_headers.emplace_back("Total_time");
  Runtime_headers.emplace_back("ND_Grouping_time");
  Runtime_headers.emplace_back("clearSubTree_time");
  Runtime_headers.emplace_back("getSubTreeOffset_time");
  Runtime_headers.emplace_back("sort_assign_time");
  Runtime_headers.emplace_back("getTreeNodeMeshs_time");
  Runtime_headers.emplace_back("findDirtyTreeNodes_time");
  Runtime_headers.emplace_back("reuse");

  PARTH::CSVManager runtime_csv(address, "some address", Runtime_headers,
                                false);

  runtime_csv.addElementToRecord(frame, "frame");
  runtime_csv.addElementToRecord(iter, "iter");
  runtime_csv.addElementToRecord(total_time, "Total_time");
  runtime_csv.addElementToRecord(ND_Grouping_time, "ND_Grouping_time");
  runtime_csv.addElementToRecord(clearSubTree_time, "clearSubTree_time");
  runtime_csv.addElementToRecord(getSubTreeOffset_time,
                                 "getSubTreeOffset_time");
  runtime_csv.addElementToRecord(sort_assign_time, "sort_assign_time");
  runtime_csv.addElementToRecord(getTreeNodeMesh_time, "getTreeNodeMeshs_time");
  runtime_csv.addElementToRecord(findDirtyTreeNodes_time,
                                 "findDirtyTreeNodes_time");
  runtime_csv.addElementToRecord(this->getReuseRatio(), "reuse");
  runtime_csv.addRecord();
  ND_Grouping_time = 0;
  clearSubTree_time = 0;
  getSubTreeOffset_time = 0;
  sort_assign_time = 0;
  getTreeNodeMesh_time = 0;
  findDirtyTreeNodes_time = 0;
  dirty_tree_node_idx.clear();
  dirty_node_idx.clear();
}

bool ND_Regions::ND_Grouping(
    int tree_node_id,   ///[in] node id
    int parent_node_id, /// <[in] parent node id
    int current_level,  ///<[in] The current level that we are dissecting
    int M_n,            ///<[in] Number of nodes in the mesh
    int *Mp,            ///<[in] The pointer array of mesh
    int *Mi,            ///<[in] The index array of mesh
    std::vector<int> &assigned_nodes, ///<[in] assigned node to this tree node
    int offset ///<[in] number of nodes before this region
) {
  if (parallel_analysis) {
    return this->ND_Grouping_parallel(tree_node_id, parent_node_id,
                                      current_level, M_n, Mp, Mi,
                                      assigned_nodes, offset);
  } else {
    return this->ND_Grouping_recursive(tree_node_id, parent_node_id,
                                       current_level, M_n, Mp, Mi,
                                       assigned_nodes, offset);
  }
}

bool ND_Regions::isAncestor(int ancestor, int node_id) {
  if (ancestor > node_id) {
    std::cerr << "isAncestor function is not called properly" << std::endl;
    return false;
  }
  int curr_ancestor = (node_id - 1) / 2;
  while (curr_ancestor > ancestor && curr_ancestor > 0) {
    curr_ancestor = (curr_ancestor - 1) / 2;
  }
  if (curr_ancestor == ancestor) {
    return true;
  }
  return false;
}

int ND_Regions::findCommonAncestor(int first, int second) {
  assert(first < num_tree_nodes);
  assert(second < num_tree_nodes);
  assert(tree_nodes[first].assigned_nodes.size() != 0);
  assert(tree_nodes[second].assigned_nodes.size() != 0);

  int higher_level_region = first;
  int lower_level_region = second;
  int lower_level = tree_nodes[lower_level_region].level;
  int higher_level = tree_nodes[higher_level_region].level;

  // Make the levels equal
  if (higher_level < lower_level) {
    higher_level_region = second;
    lower_level_region = first;
    lower_level = tree_nodes[lower_level_region].level;
    higher_level = tree_nodes[higher_level_region].level;
  }

  while (tree_nodes[higher_level_region].level !=
         tree_nodes[lower_level_region].level) {
    higher_level_region = tree_nodes[higher_level_region].parent_idx;
    assert(higher_level_region != -1);
  }
  assert(tree_nodes[higher_level_region].level ==
         tree_nodes[lower_level_region].level);

  // Find the common ancestor
  while (tree_nodes[higher_level_region].node_id !=
         tree_nodes[lower_level_region].node_id) {
    higher_level_region = tree_nodes[higher_level_region].parent_idx;
    lower_level_region = tree_nodes[lower_level_region].parent_idx;
    assert(higher_level_region != -1);
    assert(lower_level_region != -1);
  }
  assert(higher_level_region == lower_level_region);
  assert(tree_nodes[higher_level_region].level != -1);
  return higher_level_region;
}

void ND_Regions::getSepQualityPar(
    const NestedDissectionNode &node, ///<[in] Starting point in the tree
    double &sep_par,                  ///<[out] |S| \< sep_par * |Mn|
    double &child_par,                ///<[out] |L| or |R| \< child_par * |Mn|
    int &node_id ///<[out] node that these parameters are computed from
) {
  int left_size = 0;
  int right_size = 0;
  int sep_size = 0;
  double left_child_par = 0;
  double left_sep_par = 0;
  double right_child_par = 0;
  double right_sep_par = 0;
  int left_local_node_id = 0;
  int right_local_node_id = 0;

  if (node.isLeaf() ||
      getSubTreeSize(node) < 100) { // I got this 100 from Metis code
    sep_par = 0;
    child_par = 0;
    node_id = node.node_id;
    return;
  }

  if (node.left_node_idx != -1) {
    auto &left_node = this->tree_nodes[node.left_node_idx];
    left_size = getSubTreeSize(left_node);
  }
  if (node.right_node_idx != -1) {
    auto &right_node = this->tree_nodes[node.right_node_idx];
    right_size = getSubTreeSize(right_node);
  }
  sep_size = node.assigned_nodes.size();
  int local_Mn = left_size + right_size + sep_size;
  if (local_Mn < 100) {
    sep_par = 0;
    child_par = 0;
    node_id = node.node_id;
    return;
  }

  if (node.left_node_idx != -1) {
    auto &left_node = this->tree_nodes[node.left_node_idx];
    getSepQualityPar(left_node, left_sep_par, left_child_par,
                     left_local_node_id);
  }
  if (node.right_node_idx != -1) {
    auto &right_node = this->tree_nodes[node.right_node_idx];
    getSepQualityPar(right_node, right_sep_par, right_child_par,
                     right_local_node_id);
  }

  // I want the sep_par and child_par be from a single separator and
  //  I am giving more weight to the separator balance parameter (I just went
  //  with my gut)
  double tmp1 = left_size * 1.0 / local_Mn;
  double tmp2 = right_size * 1.0 / local_Mn;
  double local_sep_par = sep_size * 1.0 / local_Mn;
  double local_child_par = std::max(tmp1, tmp2);
  int local_node_id = node.node_id;

  if (local_sep_par < left_sep_par) {
    local_sep_par = left_sep_par;
    local_child_par = left_child_par;
    local_node_id = left_local_node_id;
  }

  if (local_sep_par < right_sep_par) {
    local_sep_par = right_sep_par;
    local_child_par = right_child_par;
    local_node_id = right_local_node_id;
  }
  sep_par = local_sep_par;
  child_par = local_child_par;
  node_id = local_node_id;
  return;
}

int ND_Regions::getSubTreeSize(
    const NestedDissectionNode &node ///<[in] Starting point in the tree
) {
  int size = 0;
  if (node.node_id == -1) {
    return 0;
  }

  if (node.left_node_idx != -1) {
    size += getSubTreeSize(tree_nodes[node.left_node_idx]);
  }
  if (node.right_node_idx != -1) {
    size += getSubTreeSize(tree_nodes[node.right_node_idx]);
  }
  size += node.assigned_nodes.size();
  return size;
}

void ND_Regions::updateOffset(NestedDissectionNode &node, int &base) {
  int left_size = 0;
  int right_size = 0;
  if (node.left_node_idx != -1) {
    updateOffset(tree_nodes[node.left_node_idx], base);
    left_size = getSubTreeSize(tree_nodes[node.left_node_idx]);
  }
  if (node.right_node_idx != -1) {
    int right_base = left_size + base;
    updateOffset(tree_nodes[node.right_node_idx], right_base);
    right_size = getSubTreeSize(tree_nodes[node.right_node_idx]);
  }
  node.offset = left_size + right_size + base;
}

///--------------------------------------------------------------------------
/// moveContactEdgeToCommonSeparator - Given a contact between two elements,
///--------------------------------------------------------------------------
bool ND_Regions::moveContactEdgeToCommonSeparator(
    int first, int second, std::vector<int> &invalid_common) {
  // Check whether it is a valid move
  if (invalid_common.size() != this->num_tree_nodes) {
    invalid_common.resize(num_tree_nodes);
  }

  int common_ancestor = findCommonAncestor(first, second);
  if (invalid_common[common_ancestor] == 1) {
    return false;
  }
  // Check whether all the middle separators are valid separators based on the
  // sep_par and child_par criteria during the move
  //  of a single contact point to its common_acestor
  if (validMoveToCommonSep(first, common_ancestor) &&
      validMoveToCommonSep(second, common_ancestor)) {
    // check the valid separator for the common_ancestor node
    int left_size = getSubTreeSize(tree_nodes[first]) -
                    1; // -1 is due to removing the contact point
    int right_size = getSubTreeSize(tree_nodes[right_size]) - 1;
    int sep_size = tree_nodes[common_ancestor].assigned_nodes.size() + 2;
    int local_Mn = left_size + right_size + sep_size;
    if (!(left_size * 1.0 / local_Mn < child_par &&
          right_size * 1.0 / local_Mn < child_par &&
          sep_size * 1.0 < local_Mn < sep_par)) {
      invalid_common[common_ancestor] = 1;
      return false;
    }
  } else {
    invalid_common[common_ancestor] = 1;
    return false;
  }
  // Now it is a valid move

  return true;
}

///--------------------------------------------------------------------------
/// validMoveToCommon - check whether the move is valid up to common node
///--------------------------------------------------------------------------
bool ND_Regions::validMoveToCommonSep(int &first, int &commond_id) {
  int current_region = first;
  assert(commond_id >= 0);
  assert(first >= 0);
  while (true) {
    int parent = tree_nodes[current_region].parent_idx;
    assert(parent != -1);
    if (parent != commond_id) {
      int local_Mn = getSubTreeSize(tree_nodes[parent]) - 1;
      int child_size = getSubTreeSize(tree_nodes[current_region]) - 1;
      int sep_size = tree_nodes[parent].assigned_nodes.size();
      if ((child_size * 1.0 / local_Mn < child_par) &&
          (sep_size * 1.0 / local_Mn < sep_par)) {
        current_region = parent;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }
}

///--------------------------------------------------------------------------
/// getRegionNeighbors - return the set of neighbors of a specific region
///--------------------------------------------------------------------------
void ND_Regions::getRegionNeighbors(
    int region_id,               ///<[in] input region
    std::set<int> &neighbor_list ///<[out] list of region's neighbor - the
                                 ///< regions will not be cleared
) {
  assert(Mp != nullptr);
  assert(Mi != nullptr);
  if (tree_nodes[region_id].isLeaf()) {
    auto &elements = tree_nodes[region_id].assigned_nodes;
    for (auto &elem : elements) {
      for (int sep_ptr = Mp[elem]; sep_ptr < Mp[elem + 1]; sep_ptr++) {
        int separator = Mi[sep_ptr];
        if (nodes_regions[separator] !=
            region_id) { // This neighbor is the separator
          for (int nbr_ptr = Mp[separator]; nbr_ptr < Mp[separator + 1];
               nbr_ptr++) {
            int neighbor = Mi[nbr_ptr];
            if (nodes_regions[neighbor] != region_id) {
              neighbor_list.insert(nodes_regions[neighbor]);
            }
          }
        }
      }
    }
  } else {
    auto &elements = tree_nodes[region_id].assigned_nodes;
    for (auto &elem : elements) {
      for (int nbr_ptr = Mp[elem]; nbr_ptr < Mp[elem + 1]; nbr_ptr++) {
        int neighbor = Mi[nbr_ptr];
        if (nodes_regions[neighbor] !=
            region_id) { // This neighbor is the separator
          neighbor_list.insert(nodes_regions[neighbor]);
        }
      }
    }
  }
}

} // namespace PARTH