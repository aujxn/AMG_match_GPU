#ifndef PARTITIONER_H
#define PARTITIONER_H

#include "sparse.h"

typedef struct {
  // Array with length of fine mat.nrows
  // This is where the resulting partition is stored
  int *partition;

  // Array with length of fine mat.nrows
  // Used for storing the rowsums of the strength of connection graph's
  // adjacency matrix
  double *rowsums;

  // Workspace for greedy matching, each must be the length of nnz above the
  // diagonal, i.e. length = (nnz - nrows) / 2
  int *sort_indices; // Sorting workspace and used to track who has been paired
  int *ptr_i;        // Row indices
  int *ptr_j;        // Column indices
  double *weights;   // Non-zero values
} PartitionerWorkspace;

int partition(const double *near_null, const double coarsening_factor,
              SymmCOOMatrix *symm_coo, PartitionerWorkspace *ws);

#endif // PARTITIONER_H
