//
// Created by behrooz zare on 2024-04-07.
//

#include "Integrator.h"
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <iostream>
#include <omp.h>
#include <queue>

namespace PARTH
{
  void Integrator::clearIntegrator() {
    dirty_coarse_HMD_nodes.clear();
    dirty_fine_HMD_nodes.clear();
    HMD_node_cache_flags.clear();
    dirty_coarse_HMD_nodes_saved.clear();
    dirty_fine_HMD_nodes_saved.clear();
    reuse_ratio = 1;
  }

  void Integrator::computeTheAddedRemovedEdgesForDOF(
      HMD &hmd, int dof_id, int *Mi_new, int new_start, int new_end,
      std::vector<std::pair<int, int>> &edge_set) {
    // create pairs of all edges for new mesh
    for (int i = new_start; i < new_end; i++) {
      int nbr = Mi_new[i];
      if (dof_id > nbr) {
        continue;
      }
      if (nbr == dof_id) {
        continue;
      }
      if (dof_id < nbr) {
        int condition = problematicEdgeCondition(hmd, dof_id, nbr);
        if (condition) {
          edge_set.emplace_back(dof_id, nbr);
        }
      }
    }
  }

  void Integrator::integrateDOFChanges(PARTH::HMD &hmd,
                                       std::vector<int> &new_to_old_dof_map,
                                       std::vector<int> &old_to_new_dof_map,
                                       std::vector<int> &added_dofs,
                                       std::vector<int> &removed_dofs, int M_n,
                                       int *Mp, int *Mi, int M_n_prev,
                                       std::vector<int> &mesh_perm) {
    // Compute required mapping and added and deleted DOFs
    std::vector<bool> HMD_node_with_deleted_DOFs(hmd.num_HMD_nodes, false);
    // Preprocessing the added and removed dofs for extra reuse
    // Note that in rare cases, it will reduce quality of the permutation

    for (auto &delete_dof : removed_dofs) {
      HMD_node_with_deleted_DOFs[hmd.DOF_to_HMD_node[delete_dof]] = true;
    }
    //--------------- Integrating the deleted DOFs
    std::vector<int> new_DOF_to_HMD_node(M_n, -1);
    for (auto &HMD_node : hmd.HMD_tree) {
      if (HMD_node.node_id == -1) { // Empty node
        assert(HMD_node.DOFs.empty());
        continue;
      }
      if (!HMD_node_with_deleted_DOFs[HMD_node.node_id]) { // If no deleted dof
        for (int i = 0; i < HMD_node.DOFs.size(); i++) {
          int dof = HMD_node.DOFs[i];
          int new_dof = old_to_new_dof_map[dof];
          assert(new_to_old_dof_map[new_dof] != -1);
          HMD_node.DOFs[i] = new_dof;
          new_DOF_to_HMD_node[new_dof] = HMD_node.node_id;
        }
      } else { // Integrate the deleted dof
        std::vector<int> new_DOFs;
        std::vector<int> new_permuted_label;
        std::vector<int> permutation_label_update(HMD_node.DOFs.size(), 0);
        for (int i = 0; i < HMD_node.DOFs.size(); i++) {
          int dof = HMD_node.DOFs[i];
          int new_dof = old_to_new_dof_map[dof];
          if (new_dof != -1) { // if not deleted
            new_DOFs.emplace_back(new_dof);
            new_permuted_label.emplace_back(HMD_node.permuted_new_label[i]);
            new_DOF_to_HMD_node[new_dof] = HMD_node.node_id;
          } else {
            // permuted label of deleted dof
            int label = HMD_node.permuted_new_label[i];
            for (int l = label + 1; l < HMD_node.DOFs.size(); l++) {
              permutation_label_update[l]--;
            }
          }
        }
        // Update new_permuted_new_label
        for (int i = 0; i < new_permuted_label.size(); i++) {
          new_permuted_label[i] +=
              permutation_label_update[new_permuted_label[i]];
        }
        HMD_node.DOFs = new_DOFs;
        HMD_node.permuted_new_label = new_permuted_label;
      }
      assert(HMD_node.permuted_new_label.size() == HMD_node.DOFs.size());
#ifndef NDEBUG
      std::vector<bool> valid_permute(HMD_node.permuted_new_label.size(), false);
      for (int i = 0; i < HMD_node.permuted_new_label.size(); i++) {
        assert(valid_permute[HMD_node.permuted_new_label[i]] == false);
        valid_permute[HMD_node.permuted_new_label[i]] = true;
      }
      // Check if all the valid_permute are true
      for (int i = 0; i < HMD_node.permuted_new_label.size(); i++) {
        if (!valid_permute[i]) {
          assert(false);
        }
      }
#endif
    }

    //--------------- Integrating the added DOFs
    if (!added_dofs.empty()) {

      // Find a neighbor that already belong to some sub-mesh
      std::queue<int> first_ring, second_ring;
      std::queue<int> *empty_ring;
      std::queue<int> *full_ring;
      std::queue<int> *tmp;

      // Create a random face
      full_ring = &first_ring;
      empty_ring = &second_ring;

      for (auto &dof : added_dofs) {
        (*full_ring).push(dof);
      }
      bool change_happend = true;
      while (change_happend) {
        change_happend = false;
        while (!(*full_ring).empty()) {
          int dof = (*full_ring).front();
          std::vector<int> nbr_hmd_nods;
          nbr_hmd_nods.reserve(Mp[dof + 1] - Mp[dof]);
          // Compute all the hmd nodes candidate
          for (int nbr_ptr = Mp[dof]; nbr_ptr < Mp[dof + 1]; nbr_ptr++) {
            int nbr = Mi[nbr_ptr];
            if (new_DOF_to_HMD_node[nbr] != -1) {
              nbr_hmd_nods.emplace_back(new_DOF_to_HMD_node[nbr]);
            }
          }
          // Create unique set of nodes
          std::sort(nbr_hmd_nods.begin(), nbr_hmd_nods.end());
          nbr_hmd_nods.erase(
              std::unique(nbr_hmd_nods.begin(), nbr_hmd_nods.end()),
              nbr_hmd_nods.end());
          // Find the common separators of all the possible combinations of
          // nbr_hmd_nodes edges
          if (nbr_hmd_nods.size() == 1) {
            new_DOF_to_HMD_node[dof] = nbr_hmd_nods[0];
            (*full_ring).pop();
            // Assignment happend
            change_happend = true;
            continue;
          }
          if (nbr_hmd_nods.size() == 0) {
            (*empty_ring).push(dof);
            (*full_ring).pop();
            continue;
          }

          std::vector<int> possible_hmd_nodes;
          for (int first = 0; first < nbr_hmd_nods.size(); first++) {
            for (int second = first + 1; second < nbr_hmd_nods.size(); second++) {
              int big_id = nbr_hmd_nods[first];
              int small_id = nbr_hmd_nods[second];
              if (small_id > big_id) {
                int tmp = big_id;
                big_id = small_id;
                small_id = tmp;
              }
              if (hmd.isSeparator(small_id, big_id)) {
                possible_hmd_nodes.emplace_back(big_id);
              } else {
                possible_hmd_nodes.emplace_back(
                    hmd.findCommonSeparator(small_id, big_id));
              }
            }
          }

          // Find the minimum id
          int min_id = possible_hmd_nodes[0];
          for (int i = 1; i < possible_hmd_nodes.size(); i++) {
            if (possible_hmd_nodes[i] < min_id) {
              min_id = possible_hmd_nodes[i];
            }
          }
          // assign the node to this hmd_node
          new_DOF_to_HMD_node[dof] = min_id;
          change_happend = true;
          (*full_ring).pop();
        }
        // Swap full and empty queues
        tmp = full_ring;
        full_ring = empty_ring;
        empty_ring = tmp;
      }

      // Finding a place for unassigned added new DOFs
      // Find an empty hmd_node for the assignment
      int free_HMD_node_for_unassigned_dofs = -1;
      for (int hmd_node = hmd.num_HMD_nodes - 1; hmd_node >= 0; hmd_node--) {
        if (hmd.HMD_tree[hmd_node].isLeaf() &&
            hmd.HMD_tree[hmd_node].DOFs.empty()) {
          free_HMD_node_for_unassigned_dofs = hmd_node;
          break;
            }
      }
      // If there is no empty leaf, assign to the last leaf
      if (free_HMD_node_for_unassigned_dofs == -1) {
        free_HMD_node_for_unassigned_dofs = hmd.num_HMD_nodes - 1;
      }

      // Assign to each HMD_node and add identity for the assigned node
      std::set<int> added_hmd_node;
      for (int i = 0; i < added_dofs.size(); i++) {
        int add = added_dofs[i];
        int current_HMD_node = new_DOF_to_HMD_node[add];
        if (current_HMD_node == -1) {
          new_DOF_to_HMD_node[add] = free_HMD_node_for_unassigned_dofs;
          HMD::HMDNode &hmd_node =
              hmd.HMD_tree[free_HMD_node_for_unassigned_dofs];
          int HMD_node_size = hmd_node.DOFs.size();
          hmd_node.DOFs.emplace_back(add);
          hmd_node.permuted_new_label.emplace_back(HMD_node_size);
          added_hmd_node.insert(free_HMD_node_for_unassigned_dofs);
        } else {
          HMD::HMDNode &hmd_node = hmd.HMD_tree[current_HMD_node];
          int HMD_node_size = hmd_node.DOFs.size();
          hmd_node.DOFs.emplace_back(add);
          hmd_node.permuted_new_label.emplace_back(HMD_node_size);
          added_hmd_node.insert(current_HMD_node);
        }
      }

      for (auto &node : added_hmd_node) {
        // sort the DOFs
        std::sort(hmd.HMD_tree[node].DOFs.begin(), hmd.HMD_tree[node].DOFs.end());
      }
    }
    // Updating the global variables of HMD
    hmd.updateCoarseHMDNodeOffset(0);
    hmd.DOF_to_HMD_node = new_DOF_to_HMD_node;
#ifndef NDEBUG
    for (int i = 0; i < hmd.DOF_to_HMD_node.size(); i++) {
      assert(hmd.DOF_to_HMD_node[i] != -1);
    }
#endif
    mesh_perm.clear();
    mesh_perm.resize(M_n);
    hmd.assignCoarseLocalPermToGlobal(hmd.HMD_tree[0], mesh_perm);
  }

  void Integrator::computeChangedEdges(
      HMD &hmd, int M_n, int *Mp, int *Mi, int M_n_prev, int *Mp_prev,
      int *Mi_prev, int *new_to_old_map, int *old_to_new_map,
      std::vector<std::pair<int, int>> &changed_dof_edges) {
    if (M_n_prev != M_n && new_to_old_map == nullptr) {
      changed_dof_edges.push_back(std::pair<int, int>(-1, -1));
      return;
    }

    if (new_to_old_map == nullptr) {
      for (int i = 0; i < M_n; i++) {
        int num_nbr = Mp[i + 1] - Mp[i];
        int num_nbr_prev = Mp_prev[i + 1] - Mp_prev[i];
        // If the number of neighbors are different, the node is changed
        if (num_nbr != num_nbr_prev) {
          computeTheAddedRemovedEdgesForDOF(hmd, i, Mi, Mp[i], Mp[i + 1],
                                            changed_dof_edges);
          continue;
        }
        int prev_cnt = Mp_prev[i];
        int cnt = Mp[i];
        assert(num_nbr == num_nbr_prev);
        // If the number of neighbors are the same, check the neighbors
        for (int j = 0; j < num_nbr; j++) {
          if (Mi[cnt++] != Mi_prev[prev_cnt++]) {
            computeTheAddedRemovedEdgesForDOF(hmd, i, Mi, Mp[i], Mp[i + 1],
                                              changed_dof_edges);
            break;
          }
        }
      }
    } else {
      assert(old_to_new_map != nullptr);
      assert(new_to_old_map != nullptr);

      for (int curr_dof_id = 0; curr_dof_id < M_n; curr_dof_id++) {
        int old_dof_id = new_to_old_map[curr_dof_id];
        if (old_dof_id == -1) { // It is an added dof
          // Add all the edges as new edges
          for (int i = Mp[curr_dof_id]; i < Mp[curr_dof_id + 1]; i++) {
            if (curr_dof_id < Mi[i]) {
              int condition = problematicEdgeCondition(hmd, curr_dof_id, Mi[i]);
              if (condition) {
                changed_dof_edges.push_back({curr_dof_id, Mi[i]});
              }
            }
          }
          continue;
        }
        // If the dof existed in the previous one
        int num_nbr = Mp[curr_dof_id + 1] - Mp[curr_dof_id];
        // Compute the mapped neighbours
        std::vector<int> mapped_nbrs;
        for (int i = Mp_prev[old_dof_id]; i < Mp_prev[old_dof_id + 1]; i++) {
          if (old_to_new_map[Mi_prev[i]] != -1) {
            mapped_nbrs.push_back(old_to_new_map[Mi_prev[i]]);
          }
        }
        int num_nbr_prev = mapped_nbrs.size();
        // If the number of neighbors are different, the node is changed
        if (num_nbr != num_nbr_prev) {
          computeTheAddedRemovedEdgesForDOF(hmd, curr_dof_id, Mi, Mp[curr_dof_id],
                                            Mp[curr_dof_id + 1],
                                            changed_dof_edges);
          continue;
        }
        int cnt = Mp[curr_dof_id];
        int prev_cnt = 0;
        assert(num_nbr == num_nbr_prev);
        // If the number of neighbors are the same, check the neighbors
        for (int j = 0; j < num_nbr; j++) {
          if (Mi[cnt++] != mapped_nbrs[prev_cnt++]) {
            computeTheAddedRemovedEdgesForDOF(
                hmd, curr_dof_id, Mi, Mp[curr_dof_id], Mp[curr_dof_id + 1],
                changed_dof_edges);
            break;
          }
        }
      }
    }

#ifndef NDEBUG
    for (int i = 0; i < M_n; i++) {
      int current_region = hmd.DOF_to_HMD_node[i];
      for (int nbr_ptr = Mp[i]; nbr_ptr < Mp[i + 1]; nbr_ptr++) {
        int nbr = Mi[nbr_ptr];
        int nbr_region = hmd.DOF_to_HMD_node[nbr];
        int big_id_region = current_region;
        int small_id_region = nbr_region;
        if (small_id_region > big_id_region) {
          int tmp = big_id_region;
          big_id_region = small_id_region;
          small_id_region = tmp;
        }
        if (small_id_region == big_id_region) {
          continue;
        } else if (hmd.isSeparator(small_id_region, big_id_region)) {
          continue;
        } else {
          int max_val = i;
          int min_val = nbr;
          if (max_val < min_val) {
            max_val = nbr;
            min_val = i;
          }
          std::pair<int, int> edge = {min_val, max_val};
          if (std::find(changed_dof_edges.begin(), changed_dof_edges.end(),
                        edge) == changed_dof_edges.end()) {
            assert(false);
                        }
        }
      }
    }
#endif

    return;
  }

  void Integrator::relocateDOFsForAggressiveReuse(
      HMD &hmd, int M_n, int relocation_hmd_node_id, std::vector<int> &mesh_perm,
      std::vector<std::pair<int, int>> &changed_edges,
      std::vector<int> &dirty_fine_grain_regions) {
    std::vector<std::pair<int, int>> filtered_changed_edges;
    // Step 1: Occurrence count
    std::vector<int> occurrence_count(M_n, 0);
    for (auto &edge : changed_edges) {
      occurrence_count[edge.first]++;
      occurrence_count[edge.second]++;
    }

    // Step 2: greedy algorithm to find the relocation position (hmd_node) for
    // vertices
    int NO_INIT_HMD_NODE = hmd.num_HMD_nodes + 1;
    std::vector<int> dof_relocated_hmd_node(M_n, NO_INIT_HMD_NODE);
    for (auto &edge : changed_edges) {
      // Filter the ok edges
      int &first_node = hmd.DOF_to_HMD_node[edge.first];
      int &second_node = hmd.DOF_to_HMD_node[edge.second];
      int big_id_region = first_node;
      int small_id_region = second_node;
      if (small_id_region > big_id_region) {
        int tmp = big_id_region;
        big_id_region = small_id_region;
        small_id_region = tmp;
      }
      // We only add neighbors that are not in the region tree
      if (small_id_region == big_id_region) {
        filtered_changed_edges.emplace_back(edge);
        continue;
      } else if (hmd.isSeparator(small_id_region, big_id_region)) {
        continue;
      }

      int common_separator = hmd.findCommonSeparator(first_node, second_node);
      if (common_separator < relocation_hmd_node_id) {
        int dof_id_to_relocate = -1;
        if (occurrence_count[edge.first] > occurrence_count[edge.second]) {
          dof_id_to_relocate = edge.first;
        } else {
          dof_id_to_relocate = edge.second;
        }
        if (common_separator < dof_relocated_hmd_node[dof_id_to_relocate]) {
          dof_relocated_hmd_node[dof_id_to_relocate] = common_separator;
        }
      } else {
        filtered_changed_edges.emplace_back(edge);
      }
    }
#ifndef NDEBUG
    for (auto &edge : changed_edges) {
      int first_node = hmd.DOF_to_HMD_node[edge.first];
      int second_node = hmd.DOF_to_HMD_node[edge.second];
      int common_separator = hmd.findCommonSeparator(first_node, second_node);
      if (common_separator >= 7) {
        continue;
      }
      if (dof_relocated_hmd_node[edge.first] != NO_INIT_HMD_NODE) {
        first_node = dof_relocated_hmd_node[edge.first];
      }
      if (dof_relocated_hmd_node[edge.second] != NO_INIT_HMD_NODE) {
        second_node = dof_relocated_hmd_node[edge.second];
      }
      int small_node_id = first_node;
      int big_node_id = second_node;
      if (small_node_id > big_node_id) {
        int tmp = small_node_id;
        small_node_id = big_node_id;
        big_node_id = tmp;
      }
      if (small_node_id == big_node_id) {
        continue;
      } else {
        assert(hmd.isSeparator(small_node_id, big_node_id));
      }
    }
#endif
    changed_edges = filtered_changed_edges;

    // Step 3: Relocate the DOFs
    std::vector<bool> deleted_dofs_flag(M_n, false);
    std::vector<int> added_dofs;
    std::vector<bool> HMD_node_with_deleted_DOFs(hmd.num_HMD_nodes, false);
    for (int i = 0; i < M_n; i++) {
      if (dof_relocated_hmd_node[i] != NO_INIT_HMD_NODE) {
        deleted_dofs_flag[i] = true;
        added_dofs.emplace_back(
            i); // This will be use to add these nodes to the separator
        HMD_node_with_deleted_DOFs[hmd.DOF_to_HMD_node[i]] = true;
      }
    }

    //--------------- Integrating the deleted DOFs
    for (auto &HMD_node : hmd.HMD_tree) {
      if (HMD_node.node_id == -1) { // Empty node
        assert(HMD_node.DOFs.empty());
        continue;
      }
      if (!HMD_node_with_deleted_DOFs[HMD_node.node_id]) { // If no deleted dof
        continue;
      } else { // Integrate the deleted dof
        std::vector<int> new_DOFs;
        std::vector<int> new_permuted_label;
        std::vector<int> permutation_label_update(HMD_node.DOFs.size(), 0);
        for (int i = 0; i < HMD_node.DOFs.size(); i++) {
          int dof = HMD_node.DOFs[i];
          if (!deleted_dofs_flag[dof]) { // if not deleted
            new_DOFs.emplace_back(dof);
            new_permuted_label.emplace_back(HMD_node.permuted_new_label[i]);
          } else {
            // permuted label of deleted dof
            int label = HMD_node.permuted_new_label[i];
            for (int l = label + 1; l < HMD_node.DOFs.size(); l++) {
              permutation_label_update[l]--;
            }
          }
        }
        // Update new_permuted_new_label
        for (int i = 0; i < new_permuted_label.size(); i++) {
          new_permuted_label[i] +=
              permutation_label_update[new_permuted_label[i]];
        }
        HMD_node.DOFs = new_DOFs;
        HMD_node.permuted_new_label = new_permuted_label;
      }
      assert(HMD_node.permuted_new_label.size() == HMD_node.DOFs.size());
#ifndef NDEBUG
      std::vector<bool> valid_permute(HMD_node.permuted_new_label.size(), false);
      for (int i = 0; i < HMD_node.permuted_new_label.size(); i++) {
        assert(valid_permute[HMD_node.permuted_new_label[i]] == false);
        valid_permute[HMD_node.permuted_new_label[i]] = true;
      }
      // Check if all the valid_permute are true
      for (int i = 0; i < HMD_node.permuted_new_label.size(); i++) {
        if (!valid_permute[i]) {
          assert(false);
        }
      }
#endif
    }

    //--------------- Integrating the added DOFs
    std::set<int> added_hmd_node;
    for (int i = 0; i < added_dofs.size(); i++) {
      int dof = added_dofs[i];
      int current_HMD_node = dof_relocated_hmd_node[dof];
      assert(current_HMD_node != NO_INIT_HMD_NODE);
      hmd.DOF_to_HMD_node[dof] = current_HMD_node;
      HMD::HMDNode &hmd_node = hmd.HMD_tree[current_HMD_node];
      int HMD_node_size = hmd_node.DOFs.size();
      hmd_node.DOFs.emplace_back(dof);
      hmd_node.permuted_new_label.emplace_back(HMD_node_size);
      added_hmd_node.insert(current_HMD_node);
      dirty_fine_grain_regions.emplace_back(current_HMD_node);
    }

    for (auto &node : added_hmd_node) {
      // sort the DOFs
      std::sort(hmd.HMD_tree[node].DOFs.begin(), hmd.HMD_tree[node].DOFs.end());
    }

    // Updating the global variables of HMD
    if (!added_dofs.empty()) {
      hmd.updateCoarseHMDNodeOffset(0);
      hmd.assignCoarseLocalPermToGlobal(hmd.HMD_tree[0], mesh_perm);
    }
  }

  void Integrator::getGlobalPerm(
      HMD &hmd, int M_n, int *Mp, int *Mi,
      std::vector<std::pair<int, int>> &changed_dof_edges,
      bool aggressive_reuse,
      bool lag,
      std::vector<int> &mesh_perm) {
    if (mesh_perm.empty()) {
      std::cerr << "PARTH: getGlobalPerm - mesh_perm should be allocated first"
                << std::endl;
    }

    assert(M_n != 0);
    assert(Mp != nullptr);
    assert(Mi != nullptr);
    assert(changed_dof_edges.size() != 0);

    // if the edge is -1 and -1 it means that we need full decomposition
    if (changed_dof_edges.size() == 1 && changed_dof_edges.back().first == -1 &&
        changed_dof_edges.back().second == -1) {
      dirty_coarse_HMD_nodes.clear();
      dirty_fine_HMD_nodes.clear();
      dirty_coarse_HMD_nodes.emplace_back(0);
      auto &tree_node = hmd.HMD_tree[0];
      rePartitionCoarseHMDNode(hmd, tree_node, M_n, Mp, Mi);
      // Assemble the permutation
      hmd.assignCoarseLocalPermToGlobal(tree_node, mesh_perm);
      reuse_ratio = 0;
      return;
        }

    this->dirty_coarse_HMD_nodes.clear();
    this->dirty_fine_HMD_nodes.clear();
    this->HMD_node_cache_flags.clear();
    HMD_node_cache_flags.resize(hmd.num_HMD_nodes, 1);
    if (changed_dof_edges.empty()) {
      std::cerr << "THE changes should not be empty at this point of the function"
                << std::endl;
    }

    std::vector<std::pair<int, int>> unresolved_edge_set;
    std::vector<int> edge_set_flag;
    std::vector<int> common_separator_flag;
    std::set<std::pair<int, int>> changed_hmd_edge_set;
    // TODO: Perform more analysis on these routines.
    if (aggressive_reuse) {
      relocateDOFsForAggressiveReuse(hmd, M_n, hmd.getLastHMDNodeInLevel(relocate_lvl),
                                     mesh_perm, changed_dof_edges,
                                     dirty_fine_HMD_nodes);
    }

    // Find the connection edges between regions and within regions
    findRegionConnections(hmd, changed_dof_edges, dirty_fine_HMD_nodes,
                          changed_hmd_edge_set);

    // keep unique ids

    if (aggressive_reuse) {
      std::sort(dirty_fine_HMD_nodes.begin(), dirty_fine_HMD_nodes.end());
      dirty_fine_HMD_nodes.erase(
          std::unique(dirty_fine_HMD_nodes.begin(), dirty_fine_HMD_nodes.end()),
          dirty_fine_HMD_nodes.end());
    }

    // Filter the unresolved_edge_set and dirty_fine_HMD_DOFs and find correct
    // dirty_coarse_HMD_DOFs
    for (auto &edge : changed_hmd_edge_set) {
      unresolved_edge_set.emplace_back(edge);
    }
    filterChanges(hmd, dirty_fine_HMD_nodes, dirty_coarse_HMD_nodes,
                  unresolved_edge_set);
    if(lag){
      //Bypassing the fill-reducing ordering for all the leaf nodes
      std::vector<int> tmp;
      for(int i = 0; i < dirty_fine_HMD_nodes.size(); i++){
        int node_id = dirty_fine_HMD_nodes[i];
        if(!hmd.HMD_tree[node_id].isLeaf()){
          tmp.push_back(node_id);
        }
      }
      dirty_fine_HMD_nodes = tmp;
      tmp.clear();

      //Bypassing the fill-reducing ordering for specific coarse nodes
      int lvl = hmd.getLastHMDNodeInLevel(this->lag_lvl) - 1;
      tmp.clear();
      for(int i = 0; i < dirty_coarse_HMD_nodes.size();i++){
        if(dirty_coarse_HMD_nodes[i] < lvl){
          tmp.push_back(dirty_coarse_HMD_nodes[i]);
        }
      }
      dirty_coarse_HMD_nodes = tmp;
    }


    // For debug purposes
    // TODO: Make it a debug feature later
    dirty_coarse_HMD_nodes_saved = dirty_coarse_HMD_nodes;
    dirty_fine_HMD_nodes_saved = dirty_fine_HMD_nodes;


    // If the reuse is small, do the full repartitioning
    reuse_ratio = getReuseRatio(hmd);
    if ((reuse_ratio < 0.2)) {
      dirty_fine_HMD_nodes.clear();
      dirty_coarse_HMD_nodes.clear();
      dirty_coarse_HMD_nodes.emplace_back(0);
      auto &tree_node = hmd.HMD_tree[0];
      rePartitionCoarseHMDNode(hmd, tree_node, M_n, Mp, Mi);
      // Assemble the permutation
      hmd.assignCoarseLocalPermToGlobal(tree_node, mesh_perm);
      reuse_ratio = 0;
    } else {
      // permuting fine-grain HMD nodes (no re-partitioning)
      permuteFineHMDNodes(hmd, M_n, Mp, Mi, dirty_fine_HMD_nodes, mesh_perm);

      // TODO: Parallelize this loop later
      // Fixing coarse-grain HMD nodes
      for (auto &tree_node_idx : dirty_coarse_HMD_nodes) {
        auto &tree_node = hmd.HMD_tree[tree_node_idx];
        rePartitionCoarseHMDNode(hmd, tree_node, M_n, Mp, Mi);
        // Assemble the permutation
        hmd.assignCoarseLocalPermToGlobal(tree_node, mesh_perm);
      }
    }

#ifndef NDEBUG
    // Check whether all the connection between dofs are valid
    //  based on separator and left and right sub-meshes connectivity
    for (int i = 0; i < M_n; i++) {
      for (int nbr_ptr = Mp[i]; nbr_ptr < Mp[i + 1]; nbr_ptr++) {
        int nbr = Mi[nbr_ptr];
        if (problematicEdgeCondition(hmd, i, nbr) != 0){
          assert(false);
        }
      }
    }
#endif
  }

  void Integrator::findRegionConnections(
      HMD &hmd, ///<[in] HMD decomposed submeshes
      std::vector<std::pair<int, int>>
          &changed_DOF_edges,                 ///<[in] edges changed between dofs
      std::vector<int> &dirty_fine_HMD_nodes, ///<[out] regions that has within
        ///< region contact points
      std::set<std::pair<int, int>>
          &changed_HMD_edges ///<[out] edges that show the connection between two
            ///< HMD nodes
  ) {

    if (changed_DOF_edges.empty()) {
      assert(true);
      std::cerr << "There is no change in the graph structure" << std::endl;
      return;
    }

    assert(hmd.HMD_init);
    // Find the changed edges
    std::set<int> within_region_connections;
    for (auto &dof : changed_DOF_edges) {
      int &first_region_id = hmd.DOF_to_HMD_node[dof.first];
      int &second_region_id = hmd.DOF_to_HMD_node[dof.second];
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
      } else if (hmd.isSeparator(small_id_region, big_id_region)) {
        continue;
      } else {
        changed_HMD_edges.insert({small_id_region, big_id_region});
      }
    }

    // Copying the within region connection into dirty node idx
    for (auto &iter : within_region_connections) {
      dirty_fine_HMD_nodes.emplace_back(iter);
    }
  }

  void Integrator::filterChanges(
      PARTH::HMD &hmd,              ///<[in] HMD decomposed submeshes
      std::vector<int> &dirty_fine, ///<[in/out] dirty sub-meshes id that are fine
      std::vector<int> &dirty_coarse, ///<[out] dirty sub-meshes id that are
        ///< coarse and need to be repartitioned
      std::vector<std::pair<int, int>>
          &edge_set ///<[in] an edge that shows the connection across sub-meshes
  ) {
    // given each edge, find the common ancestor
    for (auto &edge : edge_set) {
      int first_region = edge.first;
      int second_region = edge.second;
      assert(first_region != second_region);
      dirty_coarse.emplace_back(
          hmd.findCommonSeparator(first_region, second_region));
    }

    // Sort and delete the duplicates
    std::sort(dirty_coarse.begin(), dirty_coarse.end());
    dirty_coarse.erase(std::unique(dirty_coarse.begin(), dirty_coarse.end()),
                       dirty_coarse.end());

    // Clean dirty sub-meshes of fine-grain sub-meshes
    // if the dirty coarse region of their separator exists
    std::vector<int> dirty_coarse_flag(hmd.num_HMD_nodes, 0);
    std::vector<int> dirty_fine_flag(hmd.num_HMD_nodes, 0);
    for (auto &iter : dirty_coarse) {
      dirty_coarse_flag[iter] = 1;
      dirty_fine_flag[iter] = 1;
    }
    for (auto &iter : dirty_fine) {
      dirty_fine_flag[iter] = 1;
    }

    for (auto &node_idx : dirty_coarse) {
      auto &dirty_node = hmd.HMD_tree[node_idx];
      if (dirty_coarse_flag[node_idx] == 1) {
        hmd.unMarkLeftRightSubMeshes(dirty_node, dirty_coarse_flag);
        hmd.unMarkLeftRightSubMeshes(dirty_node, dirty_fine_flag);
      }
      dirty_fine_flag[node_idx] = 0;
    }

    // clean and filter the dirty_tree_node_idx
    dirty_coarse.clear();
    for (int i = 0; i < hmd.num_HMD_nodes; i++) {
      if (dirty_coarse_flag[i] == 1) {
        dirty_coarse.emplace_back(i);
      }
    }

    // clean and filter the dirty_node_idx
    std::vector<int> dirty_fine_tmp = dirty_fine;
    dirty_fine.clear();
    for (auto &iter : dirty_fine_tmp) {
      if (dirty_fine_flag[iter] == 1) {
        dirty_fine.emplace_back(iter);
      }
    }
  }

  double Integrator::getReuseRatio(PARTH::HMD &hmd) const {
    double nodes = 0;
    for (auto &dirty : dirty_coarse_HMD_nodes) {
      std::vector<int> assigned_nodes;
      hmd.getAssignedDOFsOfCoarseHMD(hmd.HMD_tree[dirty], assigned_nodes);
      nodes += assigned_nodes.size();
    }

    for (auto &dirty : dirty_fine_HMD_nodes) {
      nodes += hmd.HMD_tree[dirty].DOFs.size();
    }

    if (hmd.DOF_to_HMD_node.size() != 0) {
      return 1 - (nodes * 1.0 / hmd.DOF_to_HMD_node.size());
    } else {
      return 0.0;
    }
  }

  void Integrator::rePartitionCoarseHMDNode(HMD &hmd, HMD::HMDNode &HMD_node,
                                            int M_n, int *Mp, int *Mi) {
    if (HMD_node.node_id == 0) {
      hmd.clearCoarseHMDNode(HMD_node);
      std::vector<int> assigned_nodes(M_n);
      for (int i = 0; i < M_n; i++) {
        assigned_nodes[i] = i;
      }

      hmd.Decompose(0, -1, 0, M_n, Mp, Mi, assigned_nodes, 0);

      HMD_node_cache_flags.resize(hmd.num_HMD_nodes);
      std::fill(HMD_node_cache_flags.begin(), HMD_node_cache_flags.end(), 0);
      return;
    }

    // Get the sub-mesh to repartition
    std::vector<int> assigned_nodes;
    SubMesh local_mesh;
    hmd.getCoarseMesh(HMD_node, assigned_nodes, M_n, Mp, Mi, local_mesh);

    std::sort(assigned_nodes.begin(), assigned_nodes.end());

    int start_offset = hmd.getCoarseHMDNodeOffset(HMD_node);

    assert(start_offset >= 0 && start_offset < M_n);
    // clearSubTree should be called after start_offset
    int node_id = HMD_node.node_id;
    int parent_idx = HMD_node.parent_idx;
    int level = HMD_node.level;
    hmd.clearCoarseHMDNode(HMD_node);
    hmd.unMarkLeftRightSubMeshes(HMD_node, HMD_node_cache_flags);

    // Decompose the sub-mesh
    hmd.Decompose(node_id, parent_idx, level, local_mesh.M_n,
                  local_mesh.Mp.data(), local_mesh.Mi.data(), assigned_nodes,
                  start_offset);
  }

  void Integrator::permuteFineHMDNodes(
      HMD &hmd,
      int M_n, ///<[in] number of DOFs
      int *Mp, ///<[in] column pointer of the mesh (CSC format)
      int *Mi, ///<[in] row index of the mesh (CSC format)
      std::vector<int> &dirty_fine, std::vector<int> &mesh_perm) {

    if (dirty_fine.empty()) {
      return;
    }
    assert(!mesh_perm.empty());

    // Create the sub-meshes of each fine-grain HMD node
    std::vector<int> dirty_fine_flags(hmd.num_HMD_nodes, 0);
    for (auto &region : dirty_fine) {
      dirty_fine_flags[region] = 1;
    }
    std::vector<SubMesh> region_stack;
    hmd.createMultiSubMesh(dirty_fine_flags, hmd.DOF_to_HMD_node, M_n, Mp, Mi,
                           region_stack);

    // Fix the permutation per each node (No repartitioning)
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < dirty_fine.size(); i++) {
      // Fix the permutation
      int region_idx = dirty_fine[i];
      auto &hmd_node = hmd.HMD_tree[region_idx];
      auto &region_mesh = region_stack[region_idx];
      HMD_node_cache_flags[hmd_node.node_id] = 0;
      // Assign the permuted labels to the sep nodes
      std::vector<int> perm(region_mesh.M_n);
      double permute_time = hmd.Permute(region_mesh.M_n, region_mesh.Mp.data(), region_mesh.Mi.data(),
                  perm.data(), nullptr);
      hmd_node.setPermutedNewLabel(perm, permute_time);

      // Assign the permutation to the global node
      for (int local_node = 0; local_node < hmd_node.DOFs.size(); local_node++) {
        int global_node = hmd_node.DOFs[local_node];
        mesh_perm[hmd_node.permuted_new_label[local_node] + hmd_node.offset] =
            global_node;
      }
    }
  }

  bool Integrator::problematicEdgeCondition(HMD &hmd, int first_node,
                                           int second_node)
  {
    int &first_region_id = hmd.DOF_to_HMD_node[first_node];
    int &second_region_id = hmd.DOF_to_HMD_node[second_node];
    int big_id_region = first_region_id;
    int small_id_region = second_region_id;
    if (small_id_region > big_id_region) {
      std::swap(big_id_region, small_id_region);
    }
    // We only add neighbors that are not in the region tree
    if (small_id_region == big_id_region || hmd.isSeparator(small_id_region, big_id_region)) {
      return false;
    }
    // I have this return 1, but I have no idea why I have this. It was in the original paper
    // Submission and now I am removing it. But if it causes problem, I have to bring it back
    // But what problem?!!
    // if (small_id_region == big_id_region) {
    //   return 1;
    // } else if (hmd.isSeparator(small_id_region, big_id_region)) {
    //   return 0;
    // } else {
    //   return 2;
    // }
    // return -1;
    return true;
  }

std::vector<int> Integrator::getDirtyNodes(HMD &hmd){
  std::vector<int> total_dirty_nodes;
  //Get coarse nodes
  for(auto& hmd_node: dirty_coarse_HMD_nodes_saved){
    std::vector<int> dirty_nodes;
    hmd.getAssignedDOFsOfCoarseHMD(hmd.HMD_tree[hmd_node], dirty_nodes);
    total_dirty_nodes.insert(total_dirty_nodes.end(), dirty_nodes.begin(), dirty_nodes.end());
  }

  for(auto& hmd_node: dirty_fine_HMD_nodes_saved){
//      if(!hmd.HMD_tree[hmd_node].isLeaf()){
//          continue;
//      }
    total_dirty_nodes.insert(total_dirty_nodes.end(), hmd.HMD_tree[hmd_node].DOFs.begin(), hmd.HMD_tree[hmd_node].DOFs.end());
  }
  return total_dirty_nodes;
}

} // namespace PARTH