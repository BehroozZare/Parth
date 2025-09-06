//
// Created by behrooz zare on 2024-04-08.
//

#ifndef PARTH_SOLVER_PARTHTYPES_H
#define PARTH_SOLVER_PARTHTYPES_H

#include <vector>

namespace PARTH {
enum ReorderingType { METIS, AMD, MORTON_CODE };

///---------------------------------------------------------------------------------------\n
/// An structure that stores the submesh useful variables
///---------------------------------------------------------------------------------------\n
struct SubMesh {
public:
  bool init = false;
  int M_n = 0;
  int M_nnz = 0;
  std::vector<int> Mp;
  std::vector<int> Mi;

  std::vector<int> local_to_global_DOF_id;
  int offset = 0;

  std::vector<int> perm;

  void clear() {
    M_n = 0;
    M_nnz = 0;
    offset = 0;
    init = false;
    Mp.clear();
    Mi.clear();
    local_to_global_DOF_id.clear();
    perm.clear();
  }
};

} // namespace PARTH
#endif // PARTH_SOLVER_PARTHTYPES_H
