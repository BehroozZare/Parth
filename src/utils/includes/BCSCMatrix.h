//
// Created by george on 2020-02-15.
//

#ifndef FUSION_BCSCMATRIX_H
#define FUSION_BCSCMATRIX_H

#include "def.h"

class BCSCMatrix {
 BCSC *M;
public:
 ///\paragraph iterate over columns of the matrix, if it sees columns that can be group to a supernode
 /// it will store them in M->supernodes in the format of |1|3|4|7  where in here, columns 1,2 are a supernode and
 /// 4,5,6 are in a supernode. columns 3 is alone
 ///\param A well A is obvious :D
 ///\param limit the maximum width of supernode
 /// (for example if it is equal to 3, it will not block more than 3 columns)
 ///\param isLower Shows that the matrix is a lower matrix
 int supernodes(CSC *A, int limit, bool isLower);

 /// calculate the total number of nonzero in the matrix including zero padding
 int calcSize(CSC *A);

 ///\paragraph The create format fill the i and x vector of M
 ///Note that in here, the algorithm does a padding
 ///For example:
 ///    1
 ///    2  2
 ///    1  1  1
 ///    1  1  1
 /// will become 1 2 1 1 0 2 1 1 0 0 1 1 (Note the zero padding) in the x array
 void createFormat(CSC *A);

 // Extract a supernodal CSC from a BCSC
 CSC *compressed_BCSC_to_CSC();

 /// \paragraph Generates the BCSC arrays from A
 /// In here M->nnz will show the nnz of A with zero padding
 /// M->i and M->x are considered with zero padding
 /// M->nrows show the number of rows in a block for example
 ///    1
 ///    2  2
 ///    1  1  1
 ///    1  1  1
 /// in here the size of the block is equal to 4
 void generateBCSC(CSC *A) {
  M->nnz = calcSize(A);
  M->i = new int[M->nnz]();
  M->x = new double[M->nnz]();
  M->nrows = new int[M->nodes]();
  createFormat(A);
 }

public:
 BCSCMatrix(CSC *A) {
  M = new BCSC(A);
  M->nodes = supernodes(A, A->n, true);
  generateBCSC(A);
 }

 BCSCMatrix(CSC *A, int nodes, int *supernodes) {
  M = new BCSC(A);
  M->nodes = nodes;
  M->supernodes = new int[nodes+1]();
  std::memcpy(M->supernodes, supernodes, sizeof(int) * (nodes+1));
  generateBCSC(A);
 }


 BCSC *getBCSC() { return M; }

 ~BCSCMatrix() {
  delete(M);
 }
};


#endif //FUSION_BCSCMATRIX_H
