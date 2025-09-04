//
// Created by behrooz zare on 2024-04-07.
//

#ifndef PARTH_SOLVER_IMMOBILIZER_H
#define PARTH_SOLVER_IMMOBILIZER_H

#include "HMD.h"
#include <set>
#include <vector>

namespace PARTH {

class Integrator {
public:
  std::vector<int>
      dirty_coarse_HMD_nodes; // Coarse-grain HMD DOFs that have been changed
  std::vector<int>
      dirty_fine_HMD_nodes; // Fine-grain HMD DOFs that have been changed
  std::vector<int> HMD_node_cache_flags; // Array defining unchanged HMD nodes
  std::vector<int> dirty_coarse_HMD_nodes_saved;
  std::vector<int> dirty_fine_HMD_nodes_saved;
  double reuse_ratio = 0.0;

  //Relocate the nodes to fix edges below this lvl (included)
  int relocate_lvl = 2;
  //Lag the connection between the nodes below this lvl (excluded)
  int lag_lvl = 5;

  void clearIntegrator();

  Integrator() = default;

  ~Integrator() = default;


  ///--------------------------------------------------------------------------
  /// computeChanges - Compute the changes in the graph structure
  ///--------------------------------------------------------------------------
  void computeChangedEdges(HMD &hmd, int M_n, int *Mp, int *Mi, int M_n_prev,
                           int *Mp_prev, int *Mi_prev, int *new_to_old_map,
                           int *old_to_new_map,
                           std::vector<std::pair<int, int>> &changed_dof_edges);

  /// getGlobalPerm - This is the main function that execute Immobilizer
  /// procedure
  void getGlobalPerm(HMD &hmd, ///<[in] HMD decomposed submeshes
                     int M_n,  ///<[in] number of DOFs
                     int *Mp,  ///<[in] column pointer of the mesh (CSC format)
                     int *Mi,  ///<[in] row index of the mesh (CSC format)
                     std::vector<std::pair<int, int>>
                         &changed_dof_edges, ///<[in] problematic edges
                     bool aggressive_reuse,  ///<[in] activate aggressive reuse
                     bool lag,               ///<[in] activate lagging
                     std::vector<int> &mesh_perm ///<[out] permutation vector
  );

  /// computeTheAddedRemovedEdgesForDOF - given the adjacency list of old and
  /// new mesh, it computes the added and removed edges for a given DOF
  void
  computeTheAddedRemovedEdgesForDOF(HMD &hmd, int dof_id, int *Mi_new,
                                    int new_start, int new_end,
                                    std::vector<std::pair<int, int>> &edge_set);

  void integrateDOFChanges(PARTH::HMD &hmd,
                           std::vector<int> &new_to_old_dof_map,
                           std::vector<int> &old_to_new_dof_map,
                           std::vector<int> &added_dofs,
                           std::vector<int> &removed_dofs, int M_n, int *Mp,
                           int *Mi, int M_n_prev, std::vector<int> &mesh_perm);

  void findRegionConnections(
      HMD &hmd, ///<[in] HMD decomposed submeshes
      std::vector<std::pair<int, int>>
          &changed_DOF_edges, ///<[in] edges changed between dofs
      std::vector<int> &dirty_fine_HMD_nodes, ///<[out] regions that has within
      ///< region contact points
      std::set<std::pair<int, int>> &
          changed_HMD_edges ///<[out] edges that show the connection between two
                            ///< HMD nodes
  );

  ///--------------------------------------------------------------------------
  /// FilterChanges - Filter the changes based on overlapping repartitioning
  ///--------------------------------------------------------------------------
  void filterChanges(
      PARTH::HMD &hmd, ///<[in] HMD decomposed submeshes
      std::vector<int>
          &dirty_fine, ///<[in/out] dirty sub-meshes id that are fine
      std::vector<int> &dirty_coarse, ///<[out] dirty sub-meshes id that are
      ///< coarse and need to be repartitioned
      std::vector<std::pair<int, int>>
          &edge_set ///<[in] an edge that shows the connection across sub-meshes
  );

  ///--------------------------------------------------------------------------
  /// getReuseRatio - Compute the reuse ratio of the Parth
  ///--------------------------------------------------------------------------
  double getReuseRatio(HMD &hmd) const;

  ///--------------------------------------------------------------------------
  /// rePartitionCoarseHMDNode - Repartition the coarse HMD node
  ///--------------------------------------------------------------------------
  void rePartitionCoarseHMDNode(HMD &hmd, HMD::HMDNode &HMD_node, int M_n,
                                int *Mp, int *Mi);

  ///--------------------------------------------------------------------------
  /// rePartitionFineHMDNode - Repartition the fine-grain HMD node
  ///--------------------------------------------------------------------------
  void
  permuteFineHMDNodes(HMD &hmd,
                      int M_n, ///<[in] number of DOFs
                      int *Mp, ///<[in] column pointer of the mesh (CSC format)
                      int *Mi, ///<[in] row index of the mesh (CSC format)
                      std::vector<int> &dirty_regions_idx,
                      std::vector<int> &mesh_perm);

  ///--------------------------------------------------------------------------\n
  /// relocateDOFsForAggressiveReuse - Relocate the DOFs for aggressive reuse
  /// Any edge that has a common separator less than relocation_hmd_node_id
  /// will be resolved by using dof relocation to that common separator
  /// Note that it may have negative impact on performance if perform with large
  /// relocation_hmd_node_id\n
  ///--------------------------------------------------------------------------
  void relocateDOFsForAggressiveReuse(
      HMD &hmd,                    ///<[in] HMD decomposed submeshes
      int M_n,                     ///<[in] number of DOFs
      int relocation_hmd_node_id,  ///<[in] HMD node id threshold for relocation
      std::vector<int> &mesh_perm, ///<[in] mesh permutation to update
      std::vector<std::pair<int, int>>
          &changed_edges, ///< in/out> changed_edges
      std::vector<int>
          &dirty_fine_grain_regions ///<[out] these regions will be added to the
                                    ///<final dirty regions
  );

  ///--------------------------------------------------------------------------\n
  /// problematicEdgeCondition: return 0 for non problematic edge, 1 for within
  /// region edge, 2 for across separator edge\n
  ///--------------------------------------------------------------------------
  int problematicEdgeCondition(HMD &hmd, int first_node, int second_node);


  std::vector<int> getDirtyNodes(HMD &hmd);
};

} // namespace PARTH

#endif // PARTH_SOLVER_IMMOBILIZER_H
