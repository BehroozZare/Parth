//
// Created by behrooz on 10/11/22.
//

#ifndef PARTH_SOLVER_ND_REGIONS_H
#define PARTH_SOLVER_ND_REGIONS_H

#include "Parth_utils.h"
#include <metis.h>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>
#include "ParthTypes.h"

namespace PARTH {
class ND_Regions {
public:
    struct NestedDissectionNode {
        int left_node_idx = -1;
        int right_node_idx = -1;
        int node_id = -1;
        int parent_idx = -1;
        int offset = -1;
        int level = -1;
        std::vector<int> assigned_nodes;
        std::vector<int> permuted_new_label;
        bool identifier_is_defined = false;
        bool assigned_nodes_is_defined = false;
        bool permuted_new_label_is_defined = false;

        void clearTreeNode()
        {
            left_node_idx = -1;
            right_node_idx = -1;
            node_id = -1;
            parent_idx = -1;
            offset = -1;
            level = -1;
            assigned_nodes.clear();
            permuted_new_label.clear();
            identifier_is_defined = false;
            assigned_nodes_is_defined = false;
            permuted_new_label_is_defined = false;
        }

        void setNodeIdentifiers(const int left_node_idx, const int right_node_idx,
            const int node_id, const int parent_idx,
            const int offset, const int level)
        {
            this->left_node_idx = left_node_idx;
            this->right_node_idx = right_node_idx;
            this->node_id = node_id;
            this->parent_idx = parent_idx;
            this->offset = offset;
            this->level = level;
            this->identifier_is_defined = true;
        }

        void setAssignedNodesInitFlag() { this->assigned_nodes_is_defined = true; }

        void setPermutedNewLabel(std::vector<int>& region_perm)
        {
            assert(this->identifier_is_defined);
            permuted_new_label.resize(region_perm.size());
            for (int j = 0; j < region_perm.size(); j++) {
                permuted_new_label[region_perm[j]] = j + offset;
            }
            this->permuted_new_label_is_defined = true;
        }

        [[nodiscard]] bool treeNodeIsDefined() const
        {
            if (identifier_is_defined && assigned_nodes_is_defined && permuted_new_label_is_defined) {
                return true;
            }
            else {
                return false;
            }
            return false;
        }

        [[nodiscard]] bool treeNodeIsDefinedParallel() const
        {
            if (identifier_is_defined && assigned_nodes_is_defined && permuted_new_label_is_defined) {
                return true;
            }
            else {
                return false;
            }
            return false;
        }

        bool isLeaf() const
        {
            if (left_node_idx == -1 && right_node_idx == -1) {
                return true;
            }
            else {
                return false;
            }
        }
    };

public:
    std::vector<NestedDissectionNode> tree_nodes;
    bool regions_init = false;
    bool verbose = true;
    int num_levels = 0;
    int num_tree_nodes = 0;
    int num_contact_points = 0;
    std::vector<int> nodes_regions;
    std::vector<int> dirty_tree_node_idx;
    std::vector<int> dirty_node_idx;
    std::vector<int> dirty_tree_node_idx_saved;
    int M_n;
    int* Mp;
    int* Mi;
    std::vector<int> global_to_local_node_id;
    bool parallel_analysis;
    // Parameter that use to define good separator
    double sep_par = 0, child_par = 0;
    int sep_balance_node_id = -1;

public: // Timing variables
    double getTreeNodeMesh_time = 0;
    double sort_assign_time = 0;
    double getSubTreeOffset_time = 0;
    double clearSubTree_time = 0;
    double ND_Grouping_time = 0;
    double findDirtyTreeNodes_time = 0;

public: ///@brief for debugging purposes
    int frame;
    int iter;
    ///@brief if a region flag is 1, it is cached from previous iteration
    std::vector<int> cached_regions_flag;

public:
    ND_Regions() = default;
    ~ND_Regions() = default;
    ///--------------------------------------------------------------------------
    /// initRegions - Init the regions with METIS partitioning\n
    /// NOTE: The mesh should be in full CSC format with NO diagonal entry
    ///------------------------------ --------------------------------------------
    void initRegions(
        int num_levels, /// <[in] How many levels you want to dissect the graph
        int M_n, ///<[in] Number of elements inside the mesh
        int* Mp, ///<[in] Pointer array of mesh in CSC format
        int* Mi, ///<[in] Index array of mesh in CSC format
        std::vector<int>& mesh_perm ///<[out] The vector of permutation matrix for mesh
    );

    void assignMeshPtr(int M_n, int* Mp, int* Mi);

    //==========================================================================
    //====================== ND_Grouping Algorithms ============================
    //==========================================================================
    ///--------------------------------------------------------------------------
    /// ND_Grouping_recursive - Apply nested dissection with min-sep decomposition
    /// NOTE: this function has recursive calls
    ///------------------------------ --------------------------------------------
    bool ND_Grouping_recursive(
        int tree_node_id, ///[in] node id
        int parent_node_id, /// <[in] parent node id
        int current_level, ///<[in] The current level that we are dissecting
        int M_n, ///<[in] Number of nodes in the mesh
        int* Mp, ///<[in] The pointer array of mesh
        int* Mi, ///<[in] The index array of mesh
        std::vector<int>& assigned_nodes, ///<[in] assigned node to this tree node
        int offset ///<[in] number of nodes before this region
    );

    ///--------------------------------------------------------------------------
    /// ND_Grouping_parallel - apply nested dissection with min-sep decomposition
    /// NOTE: this function does the same thing as ND_Grouping_recursive but in
    /// parallel and it is not recursive. Also, it requires more memory
    /// TODO: create a less memory usage version of this function (It should not
    /// be necessary though)
    ///------------------------------ --------------------------------------------
    bool ND_Grouping_parallel(
        int tree_node_id, ///[in] node id
        int parent_node_id, /// <[in] parent node id
        int current_level, ///<[in] The current level that we are dissecting
        int M_n, ///<[in] Number of nodes in the mesh
        int* Mp, ///<[in] The pointer array of mesh
        int* Mi, ///<[in] The index array of mesh
        std::vector<int>& assigned_nodes, ///<[in] assigned node to this tree node
        int offset ///<[in] number of nodes before this region
    );

    //==========================================================================
    //====================== Mesh creation Algorithms ==========================
    //==========================================================================
    ///--------------------------------------------------------------------------
    /// createMultiSubMesh - given the region of nodes in a mesh, it creates
    /// submesh from those regions
    ///--------------------------------------------------------------------------
    void createMultiSubMesh(
        std::vector<int>& chosen_region_flag, ///<[in] a number of regions vector where regions
                                              ///< that should be submeshed is marked as 1
        const std::vector<int>& nodes_regions, ///<[in] nodes region
        const int& M_n, ///<[in] Number of nodes inside the full mesh
        const int* Mp, ///<[in] full mesh pointer array in CSC format
        const int* Mi, ///<[in] full mesh index array in CSC format
        std::vector<SubMesh>& regions_stack ///<[out] full mesh index array in CSC format
    );

    void
    createTriSubMesh(const std::vector<int>& nodes_regions, ///<[in] a vector of size nodes that
                                                            ///< defined the region of each node
        const int& M_n, ///<[in] Number of nodes inside the full mesh
        const int* Mp, ///<[in] full mesh pointer array in CSC format
        const int* Mi, ///<[in] full mesh index array in CSC format
        SubMesh& left_mesh, ///<[in] left submesh
        SubMesh& right_mesh, ///<[in] right_submesh
        SubMesh& sep_mesh ///<[in] separation mesh
    );

    //==========================================================================
    //=================== Assembling the permutation algorithms ================
    //==========================================================================
    ///--------------------------------------------------------------------------
    /// getGlobalPerm - Assemble the global permutation matrix from the sub
    /// regions given the contact points
    ///--------------------------------------------------------------------------
    void getGlobalPerm(std::vector<int>& contact_points,
        std::vector<int>& mesh_perm);

    //==========================================================================
    //========================= Detecting Reuse algorithms =====================
    //==========================================================================
    ///--------------------------------------------------------------------------
    /// findRegionConnections - It finds the connection between regions
    /// and within regions
    /// NOTE: The edge_set and dirty_node_idx should be filtered later
    ///--------------------------------------------------------------------------
    void findRegionConnections(
        std::vector<int>& contact_points, ///<[in] Current points with changed neighbors
        int& num_contact_points, ///<[out] number of contact points
        std::vector<int>& dirty_node_idx, ///<[out] regions that has within region
                                          ///< contact points
        std::set<std::tuple<int, int>>& edge_set ///<[out] an edge that shows the connection ///< between two
                                                 ///< regions
    );

    void filterRegionConnectios(
        std::vector<std::tuple<int, int>>& edge_set ///<[in] an edge that shows the connection ///< between
                                                    ///< two regions
    );

    ///--------------------------------------------------------------------------
    /// updateOffset - Based on the assigned node per node it will update the
    /// the subtree offset
    ///--------------------------------------------------------------------------
    void updateOffset(NestedDissectionNode& node, int& base);

    ///--------------------------------------------------------------------------
    /// moveContactEdgeToCommonSeparator - Given a contact between two elements,
    ///--------------------------------------------------------------------------
    bool moveContactEdgeToCommonSeparator(int first, int second,
        std::vector<int>& common_ancestor);

    ///--------------------------------------------------------------------------
    /// validMoveToCommon - check whether the move is valid up to common node
    ///--------------------------------------------------------------------------
    bool validMoveToCommonSep(int& first, int& commond_id);

    ///--------------------------------------------------------------------------
    /// fixSubtreePartition - recompute the permutation of the Subtree region
    /// starting from node
    ///--------------------------------------------------------------------------
    void fixSubtreePartition(NestedDissectionNode& node);

    ///--------------------------------------------------------------------------
    /// fixRegionPartition - recompute the permutation of the region represented
    /// by the node
    ///--------------------------------------------------------------------------
    void fixAndAssignRegionPermToGlobalPerm(std::vector<int>& dirty_region_idx,
        std::vector<int>& mesh_parm);

    //==========================================================================
    //================= Interaction with region tree algorithms ================
    //==========================================================================

    ///--------------------------------------------------------------------------
    /// getSepQualityPar - Measure the quality of the separator of each node
    /// and return the worst quality between nodes
    ///--------------------------------------------------------------------------
    void getSepQualityPar(
        const NestedDissectionNode& node, ///<[in] Starting point in the tree
        double& sep_par, ///<[out] |S| \< sep_par * |Mn|
        double& child_par, ///<[out] |L| or |R| \< child_par * |Mn|
        int& node_id ///<[out] node that these parameters are computed from
    );

    ///--------------------------------------------------------------------------
    /// getSubTreeSize - Return the number of nodes assigned to the subtree
    /// starting from "node" as the root
    ///--------------------------------------------------------------------------
    int getSubTreeSize(
        const NestedDissectionNode& node ///<[in] Starting point in the tree
    );

    ///--------------------------------------------------------------------------
    /// unMarkDescendent - given a node, it will make every entries corresponding
    /// to its descendant to zero. So, Marker should have values different from
    /// zero to show its effect
    ///--------------------------------------------------------------------------
    void unMarkDescendent(NestedDissectionNode& node, std::vector<int>& Marker);

    ///--------------------------------------------------------------------------
    /// unMarkDescendent - given ancestor and node_id, it will return a boolean
    /// that shows whether node_id is a descendent of ancestor or not in region
    /// structure
    ///--------------------------------------------------------------------------
    bool isAncestor(int ancestor, int node_id);

    ///--------------------------------------------------------------------------
    /// findCommonAncestor - given two nodes, it will find the common seperator of
    /// these two nodes in the regional tree
    ///--------------------------------------------------------------------------
    int findCommonAncestor(int first, int second);

    ///--------------------------------------------------------------------------
    /// getTreeNodeMesh - Given a tree node it will return the submesh of that
    /// tree node
    ///--------------------------------------------------------------------------
    void getTreeNodeMesh(
        const NestedDissectionNode& node, SubMesh& mesh,
        std::vector<int>& assigned_nodes ///<[out] Assigned nodes in this subtree
    );

    ///--------------------------------------------------------------------------
    /// createSubMeshs - create subMeshs in Lower triangular CSC format from the
    /// global vNeighbor and stores them inside regions_mesh.
    ///--------------------------------------------------------------------------
    void createSubMesh(
        const std::vector<int>& chosen_nodes, ///<[in] a vector of size nodes that
                                              ///< defined the submesh nodes
        const int& M_n, ///<[in] total number of nodes
        const int* Mp, ///<[in] global pointer vector in CSC format
        const int* Mi, ///<[in] global index vector in CSC format
        SubMesh& sub_mesh ///<[out] selected sub mesh
    );

    bool getAssignedNodeofSubtree(const NestedDissectionNode& node,
        std::vector<int>& assigned_nodes) const;

    bool getSubTreeRegions(
            const NestedDissectionNode &node, std::vector<int> &sub_tree_regions);

    std::vector<int> getDirtyRegions();

    std::vector<int> getDirtyNodes();
    ///--------------------------------------------------------------------------
    /// metisNodeNDPermutation - Find a numbering that reduces the fill-ins across
    /// regions\n NOTE: perm should be pre-allocated
    ///--------------------------------------------------------------------------
    void metisNodeNDPermutation(
        int M_n, ///<[in] Number of nodes
        int* Mp, ///<[in] Pointer array of mesh in CSC format
        int* Mi, ///<[in] index array of mesh in CSC format
        int* perm, ///<[out] a vector of size M_n with id of each node
        int* Iperm ///<[out] a vector of size M_n with reverse id
    );

    ///--------------------------------------------------------------------------
    /// assignSubtreePermToGlobal - Given a node, it assigns the permutation of
    /// the subtree starting from node to global permutation
    ///--------------------------------------------------------------------------
    bool assignSubtreePermToGlobal(NestedDissectionNode& node,
        std::vector<int>& perm);

    ///--------------------------------------------------------------------------
    /// getSubTreeOffset - given the root of the subtree find the offset (starting
    /// point) of the subtree
    /// NOTE: it is different than node offset where it is used for assembling
    /// permutation matrix
    ///--------------------------------------------------------------------------
    int getSubTreeOffset(const NestedDissectionNode& node);

    ///--------------------------------------------------------------------------
    /// clearSubTree - Clear the subtree for reinitialization
    ///--------------------------------------------------------------------------
    void clearSubTree(NestedDissectionNode& node);

    double getReuseRatio() const;

    void exportAnalysisStatistics(int frame, int iter,
        std::string analysis_name = "");
    ///--------------------------------------------------------------------------
    /// ND_Grouping - Entry function for nested dissection hierarchical grouping
    ///--------------------------------------------------------------------------
    bool ND_Grouping(
        int tree_node_id, ///<[in] node id
        int parent_node_id, ///<[in] parent node id
        int current_level, ///<[in] The current level that we are dissecting
        int M_n, ///<[in] Number of nodes in the mesh
        int* Mp, ///<[in] The pointer array of mesh
        int* Mi, ///<[in] The index array of mesh
        std::vector<int>& assigned_nodes, /// <[in] assigned node to this tree node
        int offset ///<[in] number of nodes before this region
    );

    ///--------------------------------------------------------------------------
    /// getRegionNeighbors - return the set of neighbors of a specific region
    ///--------------------------------------------------------------------------
    void getRegionNeighbors(
        int region_id, ///<[in] input region
        std::set<int>& neighbor_list ///<[out] list of region's neighbor - the
                                     ///< regions will not be cleared
    );
};
} // namespace PARTH

#endif // PARTH_SOLVER_ND_REGIONS_H
