//
// Created by behrooz on 03/11/22.
//

#ifndef PARTH_REGIONS_H
#define PARTH_REGIONS_H

#include "Parth_utils.h"
#include "metis.h"
#include <iostream>
#include <vector>

namespace PARTH {
#define BOUNDARY_REGION 2
#define CONTACT_REGION 1
#define NON_CONTACT_REGION 0

enum RegionCreationType { KWWAYS,
    THREE_REGION_CONTACT,
    MULTI_REGION_CONTACT };

class Regions {
public:
    Regions() = default;
    ~Regions() = default;

    bool regions_init = false;
    bool verbose = true;
    int num_regions = 0;
    RegionCreationType RegCType;
    std::vector<SubMesh> regions_stack;
    std::vector<SubMesh*> ordered_regions_stack;
    int num_contact_points;
    ///@brief cache variables
    std::vector<int> dirty_regions; ///@brief variables
    std::vector<int> dirty_regions_id;
    std::vector<int> nodes_regions;
    std::vector<int> prev_nodes_regions;
    std::vector<int> global_to_local_node_id;
    std::vector<int> nodes_movement_score;

    ///@brief regions connectivity in CSC format
    int CM_nnz = 0;
    std::vector<int> CMp;
    std::vector<int> CMi;

    ///@brief regions numbering
    std::vector<int> regions_perm;
    std::vector<int> regions_inv_perm;

    ///@breif 3 ways contact points variables
    double contact_region_th = 0.05; //
    double contact_region_expansion = 3;

public:
    ///--------------------------------------------------------------------------
    /// initRegions - Init the regions with METIS partitioning\n
    /// NOTE: The mesh should be in full CSC format with NO diagonal entry
    ///------------------------------ --------------------------------------------
    void
    initRegions(int num_regions,
        int& M_n, ///<[in] Number of elements inside the mesh
        int* Mp, ///<[in] Pointer array of mesh in CSC format
        int* Mi, ///<[in] Index array of mesh in CSC format
        bool partition = true ///<[in] if false it doesn't perform the partitioning
    );

    ///--------------------------------------------------------------------------
    /// metisKWaysGrouping - Init the regions with METIS partitioning
    ///------------------------------ --------------------------------------------
    bool metisKWaysGrouping(
        int num_regions, ///<[in] Number of regions to create
        int M_n, ///<[in] Number of elements
        int* Mp, ///<[in] full Mesh pointer array in CSC format
        int* Mi, ///<[in] full Mesh index array in CSC format
        std::vector<int>& nodes_regions ///<[out] For each nodes, shows its partition
    );

    ///--------------------------------------------------------------------------
    /// threeWayContactGrouping - Create a thin boundary, a contact region and a
    /// non-contact region
    ///------------------------------ --------------------------------------------
    bool threeWayContactGrouping(
        const std::vector<int>& contact_points, ///<[in] contact_points
        int M_n, ///<[in] Number of elements
        int* Mp, ///<[in] full Mesh pointer array in CSC format
        int* Mi, ///<[in] full Mesh index array in CSC format
        std::vector<int>& nodes_regions ///<[out] For each nodes, shows its partition
    );

    ///--------------------------------------------------------------------------
    /// findDirtyRegions - Find the invalid regions based on the contact points
    ///--------------------------------------------------------------------------
    void findDirtyRegions(std::vector<int>& contact_points);

    ///--------------------------------------------------------------------------
    /// fixDirtyRegions - Create new regions from dirty regions - Entry Point
    /// Function
    ///--------------------------------------------------------------------------
    void
    fixDirtyRegions(int M_n, ///<[in] Number of elements
        int* Mp, ///<[in] full Mesh pointer array in CSC format
        int* Mi, ///<[in] full Mesh index array in CSC format
        const std::vector<int>& contact_points ///<[in] contact points

    );

    ///--------------------------------------------------------------------------
    /// fixDirtyRegionsKWays - Work based on KWays metis decomposition - USELESS
    /// //TODO: Delete later
    ///--------------------------------------------------------------------------
    void
    fixDirtyRegionsKWays(int M_n, ///<[in] Number of elements
        int* Mp, ///<[in] full Mesh pointer array in CSC format
        int* Mi ///<[in] full Mesh index array in CSC format
    );

    ///--------------------------------------------------------------------------
    /// fixDirtyRegionsThreeRegionContact - Create a thin boundary region between
    /// contact area and non-contact area.
    ///--------------------------------------------------------------------------
    void fixDirtyRegionsThreeRegionContact(
        int M_n, ///<[in] Number of elements
        int* Mp, ///<[in] full Mesh pointer array in CSC format
        int* Mi, ///<[in] full Mesh index array in CSC format
        const std::vector<int>& contact_points ///<[in] contact points
    );

    ///--------------------------------------------------------------------------
    /// getGlobalPerm - Assemble the global permutation matrix from the sub
    /// regions
    ///--------------------------------------------------------------------------
    void getGlobalPerm(std::vector<int>& mesh_perm);

    ///--------------------------------------------------------------------------
    /// createSubMeshs - create full subMeshes in CSC format from the global mesh
    /// and stores it in regions_stack
    ///--------------------------------------------------------------------------
    void createMultiSubMesh(
        std::vector<int>& chosen_region_flag, ///<[in] a number of regions vector where regions
                                              ///< that should be submeshed is marked as 1
        const std::vector<int>& nodes_regions, ///<[in] a vector of size nodes that defined the
                                               ///< region of each node
        const int& M_n, ///<[in] Number of nodes inside the full mesh
        const int* Mp, ///<[in] full mesh pointer array in CSC format
        const int* Mi ///<[in] full mesh index array in CSC format
    );

    ///--------------------------------------------------------------------------
    /// createSubMeshs - create subMeshs in Lower triangular CSC format from the
    /// global vNeighbor and stores them inside regions_mesh.
    ///--------------------------------------------------------------------------
    void createSubMesh(
        const std::vector<int> chosen_nodes, ///<[in] a vector of size nodes that
                                             ///< defined the submesh nodes
        const int& M_n, ///<[in] total number of nodes
        const int* Mp, ///<[in] global pointer vector in CSC format
        const int* Mi, ///<[in] global index vector in CSC format
        SubMesh& sub_mesh ///<[out] selected sub mesh
    );

    ///--------------------------------------------------------------------------
    /// createRegionsGraph - based on the nodes' regions, create a coarsen graph
    /// that shows the connection between regions
    ///--------------------------------------------------------------------------
    void createCoarsenGraph(
        const int* Mp, ///<[in] the pointer array of coarsen graph in CSC format
        const int* Mi, ///<[in] the index array of coarsen graph in CSC format
        const std::vector<int>& nodes_regions, ///<[in] the array in which the
                                               ///< region of each node is defined
        const int& num_regions, ///<[in] Number of regions
        int& CM_nnz, ///<[out] Number of directed edges in the coarsen graph
        std::vector<int>& CMp, ///<[out] the pointer array of coarsen graph in CSC format
        std::vector<int>& CMi ///<[out] the index array of coarsen graph in CSC format
    );

    ///--------------------------------------------------------------------------
    /// printSubMeshStatistics - Print some statistics based on the elements in
    /// each regions and the status of regions
    ///--------------------------------------------------------------------------
    void printSubMeshStatistics();

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
    /// reOrderRegions - Reorder the regions and store a pointer to each of them
    /// based on a Metis_NodeND numbering for fill-in reduction across regions
    ///--------------------------------------------------------------------------
    void reOrderRegions(const int& M_n);
};

} // namespace PARTH

#endif // IPC_REGIONS_H
