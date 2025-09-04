//
// Created by behrooz zare on 2024-04-08.
//

#ifndef PARTH_SOLVER_PARTHTESTUTILS_H
#define PARTH_SOLVER_PARTHTESTUTILS_H

#include "Parth.h"
#include <string>
#include <tuple>

namespace PARTH {
class ParthTestUtils {
public:
  // Destructor nad constructor
  ParthTestUtils() = default;
  ~ParthTestUtils() = default;

  // Test Functions
  void testPermutationQuality(std::string hessian_name, std::string mesh_name);

  // ******************* Helper functions for testDeletingDOFs *****************
  double ReuseOfOne(std::string hessian_name, std::string mesh_name);
  double ReuseOfZero(std::string hessian_name, std::string mesh_name);
  bool DirtyNodeDetectionCheck(std::string hessian_name, std::string mesh_name);
  // **************** Helper functions for testAddingDeletingDOFs **************
  void testDeletingDOFs(std::string hessian_name, std::string mesh_name);
  void testAddingDOFs(std::string hessian_name, std::string mesh_name);
  static void computeTheAddedRemovedEdgesForDOF(
      int dof_id, int *Mi_new, int new_start, int new_end, int *Mi_old,
      int old_start, int old_end, std::vector<std::pair<int, int>> &edge_set);
  void testAddingDeletingEdges(std::string hessian_name, std::string mesh_name);
  void testAddingDeletingDOFs(std::string hessian_name, std::string mesh_name);

  ///--------------------------------------------------------------------------
  /// addEdgeToMesh - Add an edge to the mesh in CSC format - Both upper and
  /// lower triangular parts are presented for the mesh, thus, for symmetric
  /// mesh, make sure to have both 1->2 and 2->1 edges
  ///--------------------------------------------------------------------------
  void addEdgeToMesh(int M_n, int *Mp, int *Mi, std::vector<int> &Mp_new,
                     std::vector<int> &Mi_new,
                     std::vector<std::pair<int, int>> &edge_set);

  ///--------------------------------------------------------------------------
  /// createEdges - Create edges across DOFs. Note that to maitain symmetric
  /// meshes both 1->2 and 2->1 edges are created
  ///--------------------------------------------------------------------------
  void createEdges(
      PARTH::Parth &parth, ///<[in] Parth object
      std::vector<std::pair<int, int>>
          &HMD_edge_set,      ///<[in] Edge set in HMD nodes
      int sample_range_start, ///<[in] minimum number of DOFs edges per each
                              ///< HMD_edge_set
      int sample_range_end,   ///<[in] maximum number of DOFs edges per each
                              ///< HMD_edge_set
      std::vector<std::pair<int, int>> &edge_set ///<[out] Edge set across DOFs
  );

  ///--------------------------------------------------------------------------
  /// deleteDOF - Delete DOF from the mesh in CSC format
  ///--------------------------------------------------------------------------
  void deleteDOF(PARTH::Parth &parth, int M_n, int *Mp, int *Mi,
                 int sample_range_start, int sample_range_end,
                 std::vector<int> &Mp_new, std::vector<int> &Mi_new,
                 std::vector<int> &HMD_nodes_to_delete_dof,
                 std::vector<int> &dof_mapper);

  ///--------------------------------------------------------------------------
  /// AddDOF - Add DOF with no edge to the mesh in CSC format
  ///--------------------------------------------------------------------------
  void AddDOF(int M_n, int *Mp, int *Mi,
              int sample_range_start, ///<[in] minimum of nodes added per offset
              int sample_range_end,   ///<[in] maximum of nodes added per offset
              std::vector<int> &Mp_new, ///<[out] new mesh pointer array in CSC
              std::vector<int> &Mi_new, ///<[out] new mesh index array in CSC
              std::vector<int>
                  offset_to_add, ///<[in] starting point of the adding new dofs
              std::vector<int>
                  &dof_mapper ///<[out] mapping of the new dofs to the old dofs
  );

  ///--------------------------------------------------------------------------
  /// getLNNZ - Get the number of non-zeros in the lower triangular part of the
  /// factor L
  ///--------------------------------------------------------------------------
  int getLNNZ(int NNZ, int N, int *Ap, int *Ai, double *Ax, int *perm);
  ///--------------------------------------------------------------------------
  /// getLNNZ_AMD - Get the number of non-zeros in the lower triangular part of
  /// the factor L with AMD perm
  ///--------------------------------------------------------------------------
  int getLNNZ_AMD(int NNZ, int N, int *Ap, int *Ai, double *Ax);
  ///--------------------------------------------------------------------------
  /// getLNNZ_AMD - Get the number of non-zeros in the lower triangular part of
  /// the factor L with METIS perm
  ///--------------------------------------------------------------------------
  int getLNNZ_METIS(int NNZ, int N, int *Ap, int *Ai, double *Ax);
};
} // namespace PARTH

#endif // PARTH_SOLVER_PARTHTESTUTILS_H
