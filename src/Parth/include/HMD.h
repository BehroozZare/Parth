//
// Created by behrooz zare on 2024-04-07.
//

#ifndef PARTH_SOLVER_HMD_H
#define PARTH_SOLVER_HMD_H

#include "ParthTypes.h"
#include <cassert>
#include <vector>


namespace PARTH
{
    class HMD
    {
    public:
        struct HMDNode
        {
            int left_node_idx = -1;
            int right_node_idx = -1;
            int node_id = -1;
            int parent_idx = -1;
            int offset = -1;
            int level = -1;
            int org_size = -1;
            std::vector<int> DOFs;
            std::vector<int> permuted_new_label;
            bool identifier_is_defined = false;
            bool assigned_nodes_is_defined = false;
            bool permuted_new_label_is_defined = false;
            double separator_comp_time = 0;
            double permute_time = 0;

            void clearNode()
            {
                left_node_idx = -1;
                right_node_idx = -1;
                node_id = -1;
                parent_idx = -1;
                offset = -1;
                level = -1;
                DOFs.clear();
                permuted_new_label.clear();
                identifier_is_defined = false;
                assigned_nodes_is_defined = false;
                permuted_new_label_is_defined = false;
            }

            void setSeparatorCompTime(double separator_time){
                this->separator_comp_time = separator_time;
            }

            void setIdentifier(const int left_node_idx, const int right_node_idx,
                               const int node_id, const int parent_idx,
                               const int offset, const int level, const int org_size)
            {
                this->left_node_idx = left_node_idx;
                this->right_node_idx = right_node_idx;
                this->node_id = node_id;
                this->parent_idx = parent_idx;
                this->offset = offset;
                this->level = level;
                this->identifier_is_defined = true;
                this->org_size = org_size;
            }

            void setInitFlag() { this->assigned_nodes_is_defined = true; }

            void setPermutedNewLabel(std::vector<int>& region_perm, double time)
            {
                assert(this->identifier_is_defined);
                permuted_new_label.resize(region_perm.size());
                for (int j = 0; j < region_perm.size(); j++)
                {
                    permuted_new_label[region_perm[j]] = j;
                }
                this->permuted_new_label_is_defined = true;
                this->permute_time = time;
            }

            bool isLeaf() const
            {
                if (left_node_idx == -1 && right_node_idx == -1)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        };

    public:
        //======================== VARIABLES ========================
        bool verbose = true; // Verbose mode for debugging
        PARTH::ReorderingType reorder_type = PARTH::ReorderingType::METIS;

        std::vector<HMDNode> HMD_tree;
        bool HMD_init = false;
        int num_levels = 0;
        int num_HMD_nodes = 0;

        std::vector<int> DOF_to_HMD_node; // Shows a mapping between DOFs to HMD
        // decomposed sub-meshes

        //  SpMP::LevelSchedule *p2pScheduleForDecomposing = nullptr;
        //======================== FUNCTIONS ========================
        HMD() = default;
        ~HMD();

        void clearHMD();

        /// initHMD - Initialize the HMD tree and perform an initial decomposition
        void initHMD(
            int num_levels, /// <[in] How many levels you want to dissect the graph
            int M_n, ///<[in] Number of elements inside the mesh
            int* Mp, ///<[in] Pointer array of mesh in CSC format
            int* Mi, ///<[in] Index array of mesh in CSC format
            std::vector<int>& mesh_perm ///<[out] The mesh permutation array;
        );

        int getLastHMDNodeInLevel(int level) const;
        /// Decompose - Decompose the mesh using HMD algorithm
        bool
        Decompose(int HMD_node_id, ///[in] HMD node id
                  int HMD_parent_id, /// <[in] HMD node's parent id
                  int cur_level, ///<[in] The current level that we are dissecting
                  int M_n, ///<[in] Number of nodes in the mesh
                  int* Mp, ///<[in] The pointer array of mesh
                  int* Mi, ///<[in] The index array of mesh
                  std::vector<int>& HMD_node_DOFs,
                  ///<[in] assigned DOF to this HMD
                                                              ///< node and its descendent
                  int offset ///<[in] number of nodes before this region
        );

        /// metisNodeNDPermutation - Permute the mesh using METIS
        void
        metisNodeNDPermutation(int M_n, ///<[in] Number of DOFs in the mesh
                               int* Mp, ///<[in] Pointer array of mesh in CSC format
                               int* Mi, ///<[in] Index array of mesh in CSC format
                               int* perm, ///<[in] Permutation array
                               int* Iperm ///<[in] Inverse permutation array
        );

        /// metisNodeNDPermutation - Permute the mesh using METIS
        void
        AMDNodeNDPermutation(int M_n, ///<[in] Number of DOFs in the mesh
                             int* Mp, ///<[in] Pointer array of mesh in CSC format
                             int* Mi, ///<[in] Index array of mesh in CSC format
                             int* perm, ///<[in] Permutation array
                             int* Iperm ///<[in] Inverse permutation arrau
        );

        /// Permute - Permute the mesh
        void Permute(int M_n, ///<[in] Number of DOFs in the mesh
                     int* Mp, ///<[in] Pointer array of mesh in CSC format
                     int* Mi, ///<[in] Index array of mesh in CSC format
                     int* perm, ///<[out] Permutation array
                     int* Iperm ///<[out] Inverse permutation array
        );

        /// assignCoarseLocalPermToGlobal - Assign the coarse HMD node local
        /// permutation to the global permutation
        bool assignCoarseLocalPermToGlobal(
            HMD::HMDNode& node, ///<[in] The HMD node id
            std::vector<int>& perm ///<[in/out] The global permutation array
        );

        /// isSeparator - Check if the separator_id is a separator for node_id
        bool isSeparator(int separator_id, int node_id);

        /// findCommonSeparator - find the common separator between two nodes
        int findCommonSeparator(int first, int second);

        /// unMarkLeftRightSubMeshes - Unmark the left and right sub-meshes
        void unMarkLeftRightSubMeshes(HMDNode& node, std::vector<int>& Marker);

        /// getAssignedDOFsOfCoarseHMD - Get the assigned DOFs of the coarse HMD node
        bool getAssignedDOFsOfCoarseHMD(const HMDNode& HMD_node,
                                        std::vector<int>& assigned_nodes) const;

        /// clearCoarseHMDNode - Clear the coarse HMD nodes
        void clearCoarseHMDNode(HMDNode& node);

        /// getCoarseMesh - Get the mesh of the coarse HMD node
        void getCoarseMesh(
            const HMDNode& node, ///<[in] HMD node
            std::vector<int>& assigned_nodes,
            ///<[in] nodes that are assigned to this
                                                   ///< coarse HMD node
            int M_n, ///<[in] The size of the mesh that hmd is created based on
            int* Mp, ///<[in] The pointer array of the mesh
            int* Mi, ///<[in] the index array of the mesh
            SubMesh& mesh ///<[out] The submesh of the HMD node
        );

        /// createSubMesh - Create a sub-mesh from the HMD mesh given chosen_DOFs
        void createSubMesh(
            const std::vector<int>
            & chosen_DOFs, ///<[in] a vector of size nodes that defined the su
            const int& M_n, ///<[in] a vector of size nodes that defined the su
            const int* Mp, ///<[in] a vector of size nodes that defined the su
            const int* Mi, ///<[in] a vector of size nodes that defined the su
            SubMesh& sub_mesh);

        /// getCoarseHMDNodeOffset - Get the offset of the coarse HMD node (the offset
        /// of the left most HMD node of the current HMD node)
        int getCoarseHMDNodeOffset(const HMDNode& HMD_node);

        /// createMultiSubMesh - Create all the separated sub-meshes from a set of
        /// fine-grain HMD nodes
        void createMultiSubMesh(
            std::vector<int>
            & chosen_region_flag,
            ///<[in] a number of regions vector where regions
                 ///< that should be submeshed is marked as 1
            const std::vector<int>
            & nodes_regions, ///<[in] a vector of size nodes that
      ///< defined the region of each node
            const int& M_n, /// <[in] Number of nodes inside the full mesh
            const int* Mp, /// <[in] full mesh pointer array in CSC format
            const int* Mi, /// <[in] full mesh index array in CSC format
            std::vector<SubMesh>
            & regions_stack /// <[out] full mesh index array in CSC format
        );

        /// createTriSubMesh - Create the left, right and separator submeshes for the
        /// HMD node
        void createTriSubMesh(
            const std::vector<int>
            & nodes_regions, ///<[in] a vector of size nodes that
      ///< defined the region of each node
            const int& M_n, /// <[in] Number of nodes inside the full mesh
            const int* Mp, /// <[in] full mesh pointer array in CSC format
            const int* Mi, /// <[in] full mesh index array in CSC format
            SubMesh& left_mesh, /// <[in] left submesh
            SubMesh& right_mesh, /// <[in] right_submesh
            SubMesh& sep_mesh /// <[in] separation mesh
        );

        void updateFullOffsetRecursive(double& accumulated_offset, HMDNode& node);

        void updateCoarseHMDNodeOffset(int node_idx);

        int getTotalEdgesInHMDNode(int node_idx, int M_n, int* Mp);

        /// getCoarseNodeDOFsSize - Get the size of the DOFs assigned to the coarse HMD node
        int getCoarseNodeDOFsSize(int hmd_node_idx);

        int computeHMDLVL(int node_idx);

        /// computeP2PSchedule - This a P2P schedule based on HMD partitioning
        //  void computeP2PSchedule(int start_node, int max_level);
    };
} // namespace PARTH

#endif // PARTH_SOLVER_HMD_H
