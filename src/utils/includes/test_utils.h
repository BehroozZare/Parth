//
// Created by Kazem on 10/11/19.
//

#ifndef PROJECT_TEST_UTILS_H
#define PROJECT_TEST_UTILS_H

#include <cmath>
#include <vector>
#include "def.h"

namespace sym_lib{

 /// Generates m unique random number of the range n
 /// \param m
 /// \param n
 /// \param rand_vec
 /// \param seed
 void generate_uniq_rand_vector(int m, int n, std::vector<int> &rand_vec,
                                unsigned seed=0);

 /// Generates a random general sparse matrix with dimension of n
 /// \param n
 /// \param density
 /// \param max_val
 /// \param seed
 /// \return
 CSC *random_square_sparse(size_t n, double density, double max_val=1e4,
                         unsigned seed=1);

 CSC *random_symmetric_sparse(size_t n, double density, double max_val=1e4,
                           unsigned seed=1);

 ///
 /// \param d_min
 /// \param d_max
 /// \return
 double double_rand(double d_min, double d_max);


 /// Checks whether all values of vec is equal to val
 /// \tparam type
 /// \param n the size of input vector
 /// \param vec the input vector of size n
 /// \param val the targeted input value
 /// \return whether the vector is equal to the target value val
 template<class type>
 bool is_equal(int n, const type* vec, type val, double eps=1e-8){
  for (int i = 0; i < n; ++i) {
   if(std::abs(vec[i] - val) > eps)
    return false;
  }
  return true;
 }


 /// Checks whether vec1 is equal with vec2 in a given range.
 /// \tparam type
 /// \param beg_idx
 /// \param end_idx
 /// \param vec1
 /// \param vec2
 /// \param eps
 /// \return
 template<class type>
 bool is_equal(int beg_idx, int end_idx, const type* vec1, const type* vec2,
   double eps=1e-8){
  for (int i = beg_idx; i < end_idx; ++i) {
   if(std::isnan(vec1[i]) || std::isnan(vec2[i]))
   {
       return false;
   }

   if(std::abs(vec1[i] - vec2[i]) > eps) {
       return false;
   }
  }
  return true;
 }


 ///
 /// \param A
 /// \param B
 /// \return
 bool is_equal(CSC *A, CSC *B, double eps=1e-8);

 /// Generates a RHS for a linear solve such that the solution will be all
 /// ones where the input matrix is CSC.
 /// \param n: matrix dimension
 /// \param Ap: column pointer of input matrix
 /// \param Ai: row index of input matrix
 /// \param Ax: nonzero values
 /// \param b produced right-hand-side, should be pre-allocated with size
 ///         of n
 void rhs_init(int n, int *Ap, int *Ai, double *Ax, double *b);

 /// Generates a RHS for a linear solve such that the solution will be all
 /// ones where the input matrix is BCSC.
 /// \param n: matrix dimension
 /// \param nBlocks: number of supenodes
 /// \param Ap: column pointer
 /// \param Ai: row indices
 /// \param AiP: pointer to row indices
 /// \param Ax: nonzero values
 /// \param b produced right-hand-side, should be pre-allocated with size
 ///         of n
 void rhs_init_blocked(size_t n, size_t nBlocks, size_t *Ap, int *Ai,
                       size_t *AiP, double *Ax, double *b);


/// Checks whether there is only one unique value from range 0-n
/// in vec array otherwise it returns an error.
/// \param n
/// \param vec
/// \return
 bool test_unique(int n, int *vec);


/*
* @brief this function generates a blocked dataset that has a good load balance
* it also return the perfect schedule for this synthesize data
* @param n number of columns
* @param nnz number of non-zero elements inside the matrix
* @param part_per_core number of independent blocks that each core should compute
* @param cores number of processors to process this data
* @param Ap the pointer array in CSC format
* @param Ai the index array in CSC format
* @param Ax the values inside the array with CSC format
* @param perfect_schedule_ptr the pointer array in CSC format for nodes that going to be compute
* @param perfect_schedule_set the index array in CSC format for nodes that going to be compute
*/
 bool generateBlockedData(int n, int nnz, int part_per_core, int cores,
                          std::vector<int>& Ap, std::vector<int>& Ai, std::vector<double>& Ax,
                          std::vector<int>& perfect_schedule_ptr, std::vector<int>& perfect_schedule_set, bool apply_max_nnz = false);



 ///\Description Create a CSC version of input adjacent matrix DAG
 ///\input DAG: Adjacency matrix
 ///\input num_nodes: num of nodes in the input DAG
 ///\output Pointer to CSC version of input DAG
 CSC* generateDAGfromEdgeList(int num_nodes, std::vector<std::vector<bool>>& DAG);

 ///\Description Add a directed edge from i to j (i->j)
 ///\input_output DAG: Adjacency matrix
 ///\input i: parent of j
 ///\input j: child of i
 ///\input showWarning: if true shows warnings if anything happened
 bool addEdge(std::vector<std::vector<bool>>& DAG, int i, int j, bool showWarning = true);

}
#endif //PROJECT_TEST_UTILS_H
