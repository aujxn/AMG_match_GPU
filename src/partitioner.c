#include "partitioner.h"
#include "sparse.h"
#include <stdio.h>
#include <stdlib.h>

// Helper functions... not public API so I don't think they go in header file?
// idk I don't code in C.
void sort_weights(PartitionerWorkspace *ws, int nweights);
int pairwise_aggregation(const double coarsening_factor, const double inv_total,
                         const int fine_nrows, SymmCOOMatrix *a_bar,
                         PartitionerWorkspace *ws);

// Global variables needed for qsort...
double *global_weights;

int partition(const double *near_null, const double coarsening_factor,
              SymmCOOMatrix *symm_coo, PartitionerWorkspace *ws) {

  int nrows = symm_coo->dim;
  int nnz = symm_coo->offdiag_nnz;
  double current_cf = 1.0;

  // We start with each DOF in its own aggregate
#pragma omp parallel for
  for (int row = 0; row < nrows; row++) {
    ws->partition[row] = row;
  }

  // Calculate the row sums of the augmented matrix and the weights for the
  // strength of connection graph
  for (int k = 0; k < nnz; k++) {
    int i = symm_coo->offdiag_row_indices[k];
    int j = symm_coo->offdiag_col_indices[k];
    double val = symm_coo->offdiag_values[k];

    double weight = -val * near_null[i] * near_null[j];
    // only not thread safe part here, could parallel this...
    ws->rowsums[i] += weight;
    ws->rowsums[j] += weight;
    symm_coo->offdiag_values[k] = weight;
  }

  // Calculate the total sum of the augmented matrix and fix any negative
  // rowsums
  double inv_total = 0.0;
  for (int row = 0; row < nrows; row++) {
    // TODO negative rowsums handling.... maybe log it? idk
    double rowsum = ws->rowsums[row];
    if (rowsum < 0.0) {
      ws->rowsums[row] = 0.0;
    } else {
      inv_total += rowsum;
    }
  }
  inv_total = 1.0 / inv_total;

  double fine_nrows_float = (double)nrows;

  while (current_cf < coarsening_factor) {

    if (pairwise_aggregation(coarsening_factor, inv_total, nrows, symm_coo,
                             ws)) {
      return 1;
    }

    double coarse_nrows = (double)symm_coo->dim;
    current_cf = fine_nrows_float / coarse_nrows;
    /*
    printf("Matching step completed, cf: %.2f nrows: %d nnz: %d\n", current_cf,
           symm_coo->dim, symm_coo->dim + (symm_coo->offdiag_nnz * 2));
    */
  }

  return 0;
}

int pairwise_aggregation(const double coarsening_factor, const double inv_total,
                         const int fine_nrows, SymmCOOMatrix *a_bar,
                         PartitionerWorkspace *ws) {
  int nnz = a_bar->offdiag_nnz;
  int nrows = a_bar->dim;

  // Compute the modularity weights for the augmented graph, storing only
  // positive modularity weights
  int n_mod_weights = 0;
  for (int k = 0; k < nnz; k++) {
    int i = a_bar->offdiag_row_indices[k];
    int j = a_bar->offdiag_col_indices[k];
    double val = a_bar->offdiag_values[k];
    double mod_weight = val - inv_total * ws->rowsums[i] * ws->rowsums[j];

    if (mod_weight > 0.0) {
      ws->weights[n_mod_weights] = mod_weight;
      ws->ptr_i[n_mod_weights] = i;
      ws->ptr_j[n_mod_weights] = j;
      n_mod_weights += 1;
    }
  }

  // Cannot coarsen any more 'usefully' if all modularity weights are negative
  if (!n_mod_weights) {
    return 1;
  }

  // Sorting modularity weights makes greedy matching efficient
  sort_weights(ws, n_mod_weights);

  // Repurpose sorting array to track which DOFs have been assigned coarse
  // locations
  int *alive = ws->sort_indices;
#pragma omp parallel for
  for (int row = 0; row < nrows; row++) {
    alive[row] = -1;
  }

  // Assign match pairs to coarse grid values
  int coarse_counter = 0;
  for (int k = 0; k < n_mod_weights; k++) {
    int i = ws->ptr_i[k];
    int j = ws->ptr_j[k];
    if (alive[i] < 0 && alive[j] < 0) {
      alive[i] = coarse_counter;
      alive[j] = coarse_counter;
      coarse_counter += 1;
    }
  }

  // Assign any unmatched DOF to singleton on coarse grid
  for (int row = 0; row < nrows; row++) {
    if (alive[row] < 0) {
      alive[row] = coarse_counter;
      coarse_counter += 1;
    }
  }

  // Update the partition to reflect the assigned matching
  for (int row = 0; row < fine_nrows; row++) {
    int old_agg = ws->partition[row];
    ws->partition[row] = alive[old_agg];
  }

  // Update the augmented matrix
  int new_nnz = 0;
  for (int k = 0; k < nnz; k++) {
    int i = alive[a_bar->offdiag_row_indices[k]];
    int j = alive[a_bar->offdiag_col_indices[k]];

    if (j > i) {
      a_bar->offdiag_row_indices[new_nnz] = i;
      a_bar->offdiag_col_indices[new_nnz] = j;
      a_bar->offdiag_values[new_nnz] = a_bar->offdiag_values[k];
      new_nnz += 1;
    } else if (i > j) {
      a_bar->offdiag_row_indices[new_nnz] = j;
      a_bar->offdiag_col_indices[new_nnz] = i;
      a_bar->offdiag_values[new_nnz] = a_bar->offdiag_values[k];
      new_nnz += 1;
    }
  }
  a_bar->dim = coarse_counter;

  // Fix augmented matrix
  sum_duplicates(a_bar->offdiag_row_indices, a_bar->offdiag_col_indices,
                 a_bar->offdiag_values, ws->sort_indices, &new_nnz);
  a_bar->offdiag_nnz = new_nnz;

  // Update rowsums using weights workspace
  for (int row = 0; row < coarse_counter; row++) {
    ws->weights[row] = 0.0;
  }
  for (int row = 0; row < nrows; row++) {
    int coarse_idx = alive[row];
    ws->weights[coarse_idx] += ws->rowsums[row];
  }
  for (int row = 0; row < coarse_counter; row++) {
    ws->rowsums[row] = ws->weights[row];
  }

  return 0;
}

int compare_indices(const void *a, const void *b) {
  int idx_a = *(const int *)a;
  int idx_b = *(const int *)b;
  if (global_weights[idx_a] < global_weights[idx_b])
    return 1;
  if (global_weights[idx_a] > global_weights[idx_b])
    return -1;
  return 0;
}

void sort_weights(PartitionerWorkspace *ws, int nweights) {
  int *indices = ws->sort_indices;
  int *ptr_i = ws->ptr_i;
  int *ptr_j = ws->ptr_j;
  double *weights = ws->weights;
  global_weights = weights;

  for (int i = 0; i < nweights; i++) {
    indices[i] = i;
  }

  // TODO parallel sort is must here
  qsort(indices, nweights, sizeof(int), compare_indices);
  cycle_leader_swap(ptr_i, ptr_j, weights, indices, nweights);
}
