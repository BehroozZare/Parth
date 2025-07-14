//
// Created by behrooz on 03/11/22.
//

#include "Regions.h"

namespace PARTH {
void Regions::initRegions(int n_r, int& M_n, int* Mp, int* Mi, bool partition)
{

    RegCType = RegionCreationType::THREE_REGION_CONTACT;

    switch (RegCType) {
    case RegionCreationType::KWWAYS:
        this->num_regions = n_r;
        break;
    case RegionCreationType::THREE_REGION_CONTACT:
        this->num_regions = 3;
        break;
    default:
        std::cerr << "Unknown Region Creation Method" << std::endl;
        return;
    }

    if (this->num_regions < 2) {
        std::cerr << "PARTH: Number of regions should be more than 1" << std::endl;
        return;
    }
    if (verbose) {
        std::cout << "PARTH: Initializing Regions ..." << std::endl;
    }

    // Make all the regions dirty
    dirty_regions.resize(num_regions, 1);
    for (int i = 0; i < num_regions; i++) {
        dirty_regions_id.emplace_back(i);
    }
    ordered_regions_stack.resize(this->num_regions);
    regions_stack.resize(this->num_regions);

    // Assign nodes to regions
    std::vector<int> tmp;
    std::vector<int> chosen_regions(num_regions, 1);
    auto& non_contact_region = regions_stack[NON_CONTACT_REGION];
    nodes_regions.clear();
    if (partition) {
        switch (RegCType) {
        case RegionCreationType::KWWAYS:
            if (!metisKWaysGrouping(num_regions, M_n, Mp, Mi, nodes_regions)) {
                std::cerr << "The KWays grouping using metis was unsuccessful"
                          << std::endl;
            }
            // Create the submeshs
            createMultiSubMesh(chosen_regions, nodes_regions, M_n, Mp, Mi);
            // Create the coarsen graph that shows the connection between subregions
            createCoarsenGraph(Mp, Mi, nodes_regions, num_regions, CM_nnz, CMp, CMi);
            // order the regions
            reOrderRegions(M_n);
            break;
        case RegionCreationType::THREE_REGION_CONTACT:
            // TODO: Write the code for this part
            threeWayContactGrouping(tmp, M_n, Mp, Mi, nodes_regions);
            // Create the submeshes
            regions_stack[NON_CONTACT_REGION].clear();
            regions_stack[CONTACT_REGION].clear();
            regions_stack[BOUNDARY_REGION].clear();
            non_contact_region.M_n = M_n;
            non_contact_region.M_nnz = Mp[M_n];
            non_contact_region.Mi.resize(non_contact_region.M_nnz);
            non_contact_region.Mp.resize(non_contact_region.M_n + 1);
            std::copy(Mp, Mp + non_contact_region.M_n + 1,
                non_contact_region.Mp.data());
            std::copy(Mi, Mi + non_contact_region.M_nnz,
                non_contact_region.Mi.data());
            non_contact_region.local_to_global_DOF_id.resize(M_n);
            global_to_local_node_id.resize(M_n);
            for (int i = 0; i < M_n; i++) {
                non_contact_region.local_to_global_DOF_id[i] = i;
                global_to_local_node_id[i] = i;
            }

            // Create the coarsen graph that shows the connection between subregions
            CM_nnz = 4;
            CMp.resize(num_regions + 1);
            CMi.resize(CM_nnz);
            CMp[NON_CONTACT_REGION] = 0;
            CMp[CONTACT_REGION] = 1;
            CMp[BOUNDARY_REGION] = 2;
            CMp[BOUNDARY_REGION + 1] = 4;
            CMi[0] = BOUNDARY_REGION;
            CMi[1] = BOUNDARY_REGION;
            CMi[2] = NON_CONTACT_REGION;
            CMi[3] = NON_CONTACT_REGION;
            // Reorder the regions
            this->ordered_regions_stack.resize(this->num_regions);
            regions_perm.resize(num_regions);
            regions_inv_perm.resize(num_regions);
            for (int i = 0; i < num_regions; i++) {
                regions_perm[i] = i;
                regions_inv_perm[i] = i;
            }

            for (int i = 0; i < num_regions; i++) {
                ordered_regions_stack[i] = &regions_stack[regions_inv_perm[i]];
            }

            ordered_regions_stack[NON_CONTACT_REGION]->offset = 0;
            ordered_regions_stack[CONTACT_REGION]->offset = non_contact_region.M_n;
            ordered_regions_stack[BOUNDARY_REGION]->offset = non_contact_region.M_n;
            break;
        default:
            std::cerr << "Unknown Region Creation Method" << std::endl;
            return;
        }
    }
    regions_init = true;
}

void Regions::createSubMesh(
    const std::vector<int>
        chosen_nodes, ///<[in] a vector of size nodes that defined the su
    const int& M_n, ///<[in] a vector of size nodes that defined the su
    const int* Mp, ///<[in] a vector of size nodes that defined the su
    const int* Mi, ///<[in] a vector of size nodes that defined the su
    SubMesh& sub_mesh)
{
    if (verbose) {
        std::cout << "PARTH: createSubMesh: Create submesh of dirty regions."
                  << std::endl;
    }
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
    int total_Mn = 0;
    sub_mesh.Mp.emplace_back(sub_mesh.M_nnz);
    // Use to map the mesh nodes' id to local nodes' id
    for (int l_id = 0; l_id < sub_mesh.local_to_global_DOF_id.size(); l_id++) {
        this->global_to_local_node_id[sub_mesh.local_to_global_DOF_id[l_id]] = l_id;
    }
    // Mapping the Mi array from global ids to local ids
    for (auto& row : sub_mesh.Mi) {
        row = global_to_local_node_id[row];
    }
    assert(sub_mesh.local_to_global_node_id.size() == sub_mesh.M_n);
    assert(sub_mesh.Mp.size() == sub_mesh.M_n + 1);
    assert(sub_mesh.Mi.size() == sub_mesh.M_nnz);
    assert(sub_mesh.Mp.back() == sub_mesh.M_nnz);
}

void Regions::createMultiSubMesh(
    std::vector<int>& chosen_region_flag, ///<[in] a number of regions vector where regions
                                          ///< that should be submeshed is marked as 1
    const std::vector<int>& nodes_regions, ///<[in] a vector of size nodes that
                                           ///< defined the region of each node
    const int& M_n, /// <[in] Number of nodes inside the full mesh
    const int* Mp, /// <[in] full mesh pointer array in CSC format
    const int* Mi /// <[in] full mesh index array in CSC format
)
{
    if (verbose) {
        std::cout << "PARTH: createMultiSubMesh: Create submeshes from the regions"
                  << std::endl;
    }
    assert(nodes_regions.size() == M_n);
    assert(num_regions != 0);
    assert(Mp != nullptr);
    assert(Mi != nullptr);
    regions_stack.resize(this->num_regions);
    // Clean the chosen submeshes to be recomputed
    for (int i = 0; i < num_regions; i++) {
        if (chosen_region_flag[i] == 1) {
            regions_stack[i].clear();
        }
    }

    // Decompose the mesh
    for (int col = 0; col < M_n; col++) {
        const int& current_region = nodes_regions[col];
        // Is it in chosen regions?
        if (chosen_region_flag[current_region] == 1) {
            // Grab the region
            auto& submesh = regions_stack[current_region];
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
            auto& region = regions_stack[cnt];
            region.Mp.emplace_back(region.M_nnz);
            // Use to map the mesh nodes' id to local nodes' id
            for (int l_id = 0; l_id < region.local_to_global_DOF_id.size(); l_id++) {
                this->global_to_local_node_id[region.local_to_global_DOF_id[l_id]] = l_id;
            }
            // Mapping the Mi array from global ids to local ids
            for (auto& row : region.Mi) {
                row = global_to_local_node_id[row];
            }
            assert(region.local_to_global_node_id.size() == region.M_n);
            assert(region.Mp.size() == region.M_n + 1);
            assert(region.Mi.size() == region.M_nnz);
            assert(region.Mp.back() == region.M_nnz);
        }
    }
}

void Regions::createCoarsenGraph(
    const int* Mp, ///<[in] the pointer array of coarsen graph in CSC format
    const int* Mi, ///<[in] the index array of coarsen graph in CSC format
    const std::vector<int>& nodes_regions, ///<[in] the array in which the
                                           ///< region of each node is defined
    const int& num_regions, ///<[in] Number of regions
    int& CM_nnz, ///<[out] Number of directed edges in the coarsen graph
    std::vector<int>& CMp, ///<[out] the pointer array of coarsen graph in CSC format
    std::vector<int>& CMi ///<[out] the index array of coarsen graph in CSC format
)
{
    assert(Mp != nullptr);
    assert(Mi != nullptr);
    assert(!nodes_regions.empty());
    if (verbose) {
        std::cout << "PARTH: createCoarsenGraph: Create the graph of connectivity "
                     "of the regions"
                  << std::endl;
    }
    CMp.clear();
    CMi.clear();
    CM_nnz = 0;

    std::vector<std::vector<int>> DAG(num_regions);

#pragma omp parallel for reduction(+ \
                                   : CM_nnz)
    for (int g = 0; g < regions_stack.size(); g++) {
        auto& node_set = regions_stack[g].local_to_global_DOF_id;
        for (auto& node : node_set) {
            for (int child_ptr = Mp[node]; child_ptr < Mp[node + 1]; child_ptr++) {
                int child = nodes_regions[Mi[child_ptr]];
                if (child != nodes_regions[node]) {
                    DAG[g].push_back(child);
                }
            }
        }
        std::sort(DAG[g].begin(), DAG[g].end());
        DAG[g].erase(std::unique(DAG[g].begin(), DAG[g].end()), DAG[g].end());
        CM_nnz += DAG[g].size();
    }

    CMp.resize(num_regions + 1, 0);
    CMi.resize(CM_nnz);
    long int cti, edges = 0;
    for (cti = 0, edges = 0; cti < num_regions; cti++) {
        CMp[cti] = edges;
        for (int ctj = 0; ctj < DAG[cti].size(); ctj++) {
            CMi[edges++] = DAG[cti][ctj];
        }
    }
    assert(CM_nnz == edges);
    CMp[cti] = edges;
}

void Regions::reOrderRegions(const int& M_n)
{
    if (verbose) {
        std::cout << "PARTH: Reordering the regions" << std::endl;
    }
    this->ordered_regions_stack.resize(this->num_regions);
    regions_perm.resize(num_regions);
    regions_inv_perm.resize(num_regions);
    metisNodeNDPermutation(num_regions, CMp.data(), CMi.data(),
        regions_perm.data(), regions_inv_perm.data());
    for (int i = 0; i < num_regions; i++) {
        ordered_regions_stack[i] = &regions_stack[regions_inv_perm[i]];
    }

    int offset = 0;
    for (auto& iter : ordered_regions_stack) {
        iter->offset = offset;
        offset += iter->M_n;
    }
    assert(offset == M_n);

#ifndef NDEBUG
    std::vector<bool> node_flag(M_n, false);
    for (auto& iter : regions_stack) {
        for (auto& global_node : iter.local_to_global_node_id) {
            node_flag[global_node] = true;
        }
    }

    for (int i = 0; i < M_n; i++) {
        if (!node_flag[i]) {
            assert(false);
        }
    }
#endif
}

void Regions::metisNodeNDPermutation(int M_n, int* Mp, int* Mi, int* perm,
    int* Iperm)
{
    idx_t N = M_n;
    idx_t NNZ = Mp[M_n];
    if (NNZ == 0) {
        for (int i = 0; i < M_n; i++) {
            std::cout << "WARNING: SOME OF THE REGIONS ARE CONSIST EMPTY NODES"
                      << std::endl;
            perm[i] = i;
        }
        return;
    }
    // TODO add memory allocation protection later like CHOLMOD
    if (Iperm == nullptr) {
        std::vector<int> tmp(M_n);
        METIS_NodeND(&N, Mp, Mi, NULL, NULL, perm, tmp.data());
    }
    else {
        METIS_NodeND(&N, Mp, Mi, NULL, NULL, perm, Iperm);
    }
}

void Regions::findDirtyRegions(std::vector<int>& contact_points)
{
    num_contact_points = contact_points.size();
    dirty_regions.clear();
    dirty_regions.resize(num_regions, 0);
    for (auto& contact_point : contact_points) {
        dirty_regions[nodes_regions[contact_point]] = 1;
    }
}
void Regions::fixDirtyRegions(int M_n, int* Mp, int* Mi,
    const std::vector<int>& contact_points)
{

    switch (RegCType) {
    case RegionCreationType::KWWAYS:
        fixDirtyRegionsKWays(M_n, Mp, Mi);
        break;
    case RegionCreationType::THREE_REGION_CONTACT:
        fixDirtyRegionsThreeRegionContact(M_n, Mp, Mi, contact_points);
        break;
    default:
        std::cerr << "Unknown Region Creation Method" << std::endl;
        return;
    }
}
void Regions::fixDirtyRegionsKWays(int M_n, int* Mp, int* Mi)
{
    // Count and store the id of dirty regions
    dirty_regions_id.clear();
    for (int i = 0; i < num_regions; i++) {
        if (dirty_regions[i] == 1) {
            dirty_regions_id.emplace_back(i);
        }
    }
    // TODO: fix the situation where only a single region changes.
    if (dirty_regions_id.size() < 2) {
        return;
    }
    // Mark the nodes that should be reassigned to regions
    std::vector<int> chosen_nodes(M_n, 0);
    for (auto& iter : dirty_regions_id) {
        auto& region = regions_stack[iter];
        for (int local_node = 0; local_node < region.M_n; local_node++) {
            chosen_nodes[region.local_to_global_DOF_id[local_node]] = 1;
        }
    }

    // Find the mesh of the regions
    SubMesh dirty_regions_mesh;
    createSubMesh(chosen_nodes, M_n, Mp, Mi, dirty_regions_mesh);

    // Create new X number of regions where X is equal to the number of dirty
    // regions
    std::vector<int> dirty_submesh_nodes_regions;
    metisKWaysGrouping(dirty_regions_id.size(), dirty_regions_mesh.M_n,
        dirty_regions_mesh.Mp.data(), dirty_regions_mesh.Mi.data(),
        dirty_submesh_nodes_regions);

    // Assign the new regions to the mesh stack
    // Basically we have local regions from the dirty submesh and
    // local nodes from the dirty mesh. We are mapping the local nodes and regions
    // to their global counter parts
    for (int i = 0; i < dirty_regions_mesh.local_to_global_DOF_id.size(); i++) {
        nodes_regions[dirty_regions_mesh.local_to_global_DOF_id[i]] = dirty_regions_id[dirty_submesh_nodes_regions[i]];
    }
    // Re-assign the new regions and replace the previous dirty regions
    createMultiSubMesh(dirty_regions, nodes_regions, M_n, Mp, Mi);
    // Create the coarsen graph that shows the connection between subregions
    createCoarsenGraph(Mp, Mi, nodes_regions, num_regions, CM_nnz, CMp, CMi);
    // order the regions
    reOrderRegions(M_n);
    prev_nodes_regions = nodes_regions;
}
void Regions::fixDirtyRegionsThreeRegionContact(
    int M_n, int* Mp, int* Mi, const std::vector<int>& contact_points)
{
    // Count and store the id of dirty regions
    dirty_regions_id.clear();

    if (dirty_regions[NON_CONTACT_REGION] == 1 || dirty_regions[BOUNDARY_REGION] == 1) { // TODO: Maybe you can change this boundary condition (it is not
                                                                                         // that important I suppose)
        dirty_regions.clear();
        dirty_regions.resize(3, 1); // Mark all regions as dirty
        threeWayContactGrouping(contact_points, M_n, Mp, Mi, nodes_regions);
    }

    for (int i = 0; i < num_regions; i++) {
        if (dirty_regions[i] == 1) {
            dirty_regions_id.emplace_back(i);
        }
    }
    createMultiSubMesh(dirty_regions, nodes_regions, M_n, Mp, Mi);

#ifndef NDEBUG
    // Check the mapping
    std::vector<bool> total_mapping(M_n, false);
    for (auto& iter : regions_stack) {
        for (int local_node = 0; local_node < iter.M_n; local_node++) {
            total_mapping[iter.local_to_global_node_id[local_node]] = true;
        }
    }

    for (int i = 0; i < total_mapping.size(); i++) {
        if (!total_mapping[i]) {
            assert(false);
        }
    }

    // Check the neighbors
    auto& contact_region = regions_stack[CONTACT_REGION];
    for (int i = 0; i < contact_region.M_n; i++) {
        int global_node = contact_region.local_to_global_node_id[i];
        for (int nbr_ptr = Mp[global_node]; nbr_ptr < Mp[global_node + 1];
             nbr_ptr++) {
            int nbr = Mi[nbr_ptr];
            if (nodes_regions[nbr] == NON_CONTACT_REGION) {
                assert(false);
            }
        }
    }

    auto& non_contact_region = regions_stack[NON_CONTACT_REGION];
    for (int i = 0; i < non_contact_region.M_n; i++) {
        int global_node = non_contact_region.local_to_global_node_id[i];
        for (int nbr_ptr = Mp[global_node]; nbr_ptr < Mp[global_node + 1];
             nbr_ptr++) {
            int nbr = Mi[nbr_ptr];
            if (nodes_regions[nbr] == CONTACT_REGION) {
                assert(false);
            }
        }
    }

#endif
    // Create the coarsen graph that shows the connection between subregions
    CM_nnz = 4;
    CMp.resize(num_regions + 1);
    CMi.resize(CM_nnz);
    CMp[NON_CONTACT_REGION] = 0;
    CMp[CONTACT_REGION] = 1;
    CMp[BOUNDARY_REGION] = 2;
    CMp[BOUNDARY_REGION + 1] = 4;
    CMi[0] = BOUNDARY_REGION;
    CMi[1] = BOUNDARY_REGION;
    CMi[2] = NON_CONTACT_REGION;
    CMi[3] = NON_CONTACT_REGION;
    // Reorder the regions
    this->ordered_regions_stack.resize(this->num_regions);
    regions_perm.resize(num_regions);
    regions_inv_perm.resize(num_regions);
    for (int i = 0; i < num_regions; i++) {
        regions_perm[i] = i;
        regions_inv_perm[i] = i;
    }

    for (int i = 0; i < num_regions; i++) {
        ordered_regions_stack[i] = &regions_stack[regions_inv_perm[i]];
    }

    int offset = 0;
    for (auto& iter : ordered_regions_stack) {
        iter->offset = offset;
        offset += iter->M_n;
    }
    assert(offset == M_n);
    // TODO: You can maybe make this faster
    prev_nodes_regions = nodes_regions;
}
void Regions::getGlobalPerm(std::vector<int>& mesh_perm)
{
    assert(num_regions == dirty_regions.size());
    if (mesh_perm.empty()) {
        std::cerr << "PARTH: getGlobalPerm - mesh_perm should be allocated first"
                  << std::endl;
    }

    // TODO: Parallelize this lsoop
    for (int i = 0; i < dirty_regions_id.size(); i++) {
        auto& region = regions_stack[dirty_regions_id[i]];
        if (region.M_n != 0) { // If region is not empty
            region.perm.resize(region.M_n);
            metisNodeNDPermutation(region.M_n, region.Mp.data(), region.Mi.data(),
                region.perm.data(), NULL);

            std::vector<int> local_new_label(region.M_n);
            for (int j = 0; j < region.perm.size(); j++) {
                local_new_label[region.perm[j]] = j;
            }
            for (int local_node = 0;
                 local_node < region.local_to_global_DOF_id.size(); local_node++) {
                int global_node = region.local_to_global_DOF_id[local_node];
                mesh_perm[local_new_label[local_node] + region.offset] = global_node;
            }
        }
    }
}

bool Regions::metisKWaysGrouping(
    int num_regions, ///<[in] Number of regions to create
    int M_n, ///<[in] Number of elements
    int* Mp, ///<[in] full Mesh pointer array in CSC format
    int* Mi, ///<[in] full Mesh index array in CSC format
    std::vector<int>& nodes_regions ///<[out] For each nodes, shows its partition
)
{
    assert(M_n != 0);
    if (verbose) {
        std::cout << "PARTH: Using Metis Kways Grouping" << std::endl;
    }
    idx_t nVertices = M_n;
    idx_t nWeights = 1;
    idx_t nParts = num_regions;
    idx_t objval;
    nodes_regions.resize(nVertices, 0);
    std::vector<int> vweight(nVertices, 1);

    // TODO: Maybe Metis_PartGraphKways changes the Mp and Mi. Check that later
    // TODO: The matrix does not have diagonal values -> check the partition
    // quality
    // TODO: The PartGraphKways does not guarantee connected partitions fix that
    // too.
    int ret = METIS_PartGraphKway(&M_n, &nWeights, Mp, Mi, NULL, &nVertices, NULL,
        &nParts, NULL, NULL, NULL, &objval,
        nodes_regions.data());
    return (ret == METIS_OK);
}

bool Regions::threeWayContactGrouping(
    const std::vector<int>& contact_points, ///<[in] Contact Points
    int M_n, ///<[in] Number of elements
    int* Mp, ///<[in] full Mesh pointer array in CSC format
    int* Mi, ///<[in] full Mesh index array in CSC format
    std::vector<int>& nodes_regions ///<[out] For each nodes, shows its partition
)
{
    // contact region grouping
    nodes_regions.clear();
    nodes_regions.resize(M_n, NON_CONTACT_REGION);
    if (contact_points.empty()) {
        return true;
    }

    // Expand the contact point region for enablaing the reuse
    std::vector<int> tmp;
    std::vector<int> boundary_contact_points = contact_points;
    double num_contacts = contact_points.size();
    double num_nodes_in_contact_region = 0;
    while (((num_nodes_in_contact_region / num_contacts) < contact_region_expansion) || ((num_nodes_in_contact_region / M_n) < contact_region_th)) {
        for (auto& contact_point : boundary_contact_points) {
            nodes_regions[contact_point] = CONTACT_REGION;
            for (int nbr_ptr = Mp[contact_point]; nbr_ptr < Mp[contact_point + 1];
                 nbr_ptr++) {
                int neighbor = Mi[nbr_ptr];
                if (nodes_regions[neighbor] != CONTACT_REGION) {
                    nodes_regions[neighbor] = CONTACT_REGION;
                    tmp.emplace_back(neighbor);
                    num_nodes_in_contact_region++;
                }
            }
        }
        boundary_contact_points = tmp;
    }

    // Find boundary nodes
#pragma omp parallel for
    for (int node = 0; node < M_n; node++) {
        if (nodes_regions[node] == CONTACT_REGION) {
            bool has_non_contact_nbr = false;
            bool has_contact_nbr = false;
            for (int nbr_ptr = Mp[node]; nbr_ptr < Mp[node + 1]; nbr_ptr++) {
                int neighbor = Mi[nbr_ptr];
                if (nodes_regions[neighbor] == CONTACT_REGION) {
                    has_contact_nbr = true;
                }
                if (nodes_regions[neighbor] == NON_CONTACT_REGION) {
                    has_non_contact_nbr = true;
                }
            }
            if (has_non_contact_nbr && has_contact_nbr) {
                nodes_regions[node] = BOUNDARY_REGION;
            }
        }
        else {
            continue;
        }
    }

    return true;
}

void Regions::printSubMeshStatistics()
{
    if (verbose) {
        double nodes = 0;
        for (int i = 0; i < num_regions; i++) {
            if (dirty_regions[i] == 0) {
                nodes += regions_stack[i].M_n;
            }
        }
        if (num_regions != 0) {
            std::cout << "+++ PARTH: The percentage of reused: "
                      << (nodes * 1.0 / nodes_regions.size()) * 100 << std::endl;
        }
        if (this->RegCType == RegionCreationType::THREE_REGION_CONTACT) {
            std::cout << "+++ PARTH: The percentage of nodes in contact regions: "
                      << (regions_stack[CONTACT_REGION].M_n * 1.0 / nodes_regions.size()) * 100
                      << std::endl;
            std::cout << "+++ PARTH: The percentage of nodes in boundary regions: "
                      << (regions_stack[BOUNDARY_REGION].M_n * 1.0 / nodes_regions.size()) * 100
                      << std::endl;
            std::cout << "+++ PARTH: The percentage of nodes in non-contact regions: "
                      << (regions_stack[NON_CONTACT_REGION].M_n * 1.0 / nodes_regions.size()) * 100
                      << std::endl;
        }
        if (!nodes_regions.empty()) {
            std::cout << "+++ PARTH: The percentage of contact points: "
                      << (num_contact_points * 1.0 / nodes_regions.size()) * 100
                      << std::endl;
        }
    }
}

} // namespace PARTH
