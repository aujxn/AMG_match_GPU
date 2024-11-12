#include "mg_builder.h"
#include "interpolation.h"
#include "partitioner.h"
#include "sparse.h"

int builder(const double coarsening_factor, SymmCOOMatrix *fine_mat,
            PWCHierarchy *hierarchy) {

  int max_levels = 1;
  double size = 1;
  while (size < (double)fine_mat->dim) {
    size *= coarsening_factor;
    max_levels++;
  }

  int dim = fine_mat->dim;
  double *near_null = (double *)malloc(dim * sizeof(double));

  for (int i = 0; i < dim; i++) {
    near_null[i] = 1.0;
  }

  PiecewiseConstantTransfer **transer_operators =
      (PiecewiseConstantTransfer **)malloc(max_levels *
                                           sizeof(PiecewiseConstantTransfer *));
  SymmCOOMatrix **matrices =
      (SymmCOOMatrix **)malloc(max_levels * sizeof(SymmCOOMatrix *));

  matrices[0] = fine_mat;

  int current_dim = fine_mat->dim;
  int levels = 1;

  PartitionerWorkspace ws;
  int nweights = fine_mat->offdiag_nnz;
  ws.partition = (int *)malloc((fine_mat->dim) * sizeof(int));
  ws.rowsums = (double *)malloc((fine_mat->dim) * sizeof(double));
  ws.ptr_i = (int *)malloc(nweights * sizeof(int));
  ws.ptr_j = (int *)malloc(nweights * sizeof(int));
  ws.weights = (double *)malloc(nweights * sizeof(double));
  ws.sort_indices = (int *)malloc(nweights * sizeof(int));

  SymmCOOMatrix prev;
  // Don't need `prev.diag` for partitioner so don't allocate it
  prev.offdiag_row_indices = (int *)malloc(nweights * sizeof(int));
  prev.offdiag_col_indices = (int *)malloc(nweights * sizeof(int));
  prev.offdiag_values = (double *)malloc(nweights * sizeof(double));

  while (current_dim > 100) {
    prev.dim = current_dim;
    prev.offdiag_nnz = matrices[levels - 1]->offdiag_nnz;

    for (int k = 0; k < prev.offdiag_nnz; k++) {
      prev.offdiag_row_indices[k] =
          matrices[levels - 1]->offdiag_row_indices[k];
      prev.offdiag_col_indices[k] =
          matrices[levels - 1]->offdiag_col_indices[k];
      prev.offdiag_values[k] = matrices[levels - 1]->offdiag_values[k];
    }

    int failure = partition(near_null, coarsening_factor, &prev, &ws);
    if (failure) {
      break;
    }

    SymmCOOMatrix *a_coarse = (SymmCOOMatrix *)malloc(sizeof(SymmCOOMatrix));
    PiecewiseConstantTransfer *p =
        (PiecewiseConstantTransfer *)malloc(sizeof(PiecewiseConstantTransfer));

    p->fine_dim = current_dim;
    p->coarse_dim = prev.dim;
    p->partition = (int *)malloc(current_dim * sizeof(int));
    p->weights = (double *)malloc(current_dim * sizeof(double));

    for (int k = 0; k < current_dim; k++) {
      p->partition[k] = ws.partition[k];
      p->weights[k] = near_null[k];
    }

    coarsen(matrices[levels - 1], p, a_coarse);
    matrices[levels] = a_coarse;
    transer_operators[levels - 1] = p;

    double resulting_cf = ((double)current_dim) / ((double)p->coarse_dim);
    levels++;
    printf("Added level %d\n\tcf: %.2f nrows: %d nnz: %d\n", levels,
           resulting_cf, prev.dim, prev.dim + (prev.offdiag_nnz * 2));
    current_dim = p->coarse_dim;
  }

  hierarchy->coarsening_factor = coarsening_factor;
  hierarchy->levels = levels;
  hierarchy->matrices = matrices;
  hierarchy->transer_operators = transer_operators;

  free(ws.partition);
  free(ws.rowsums);
  free(ws.ptr_i);
  free(ws.ptr_j);
  free(ws.weights);
  free(ws.sort_indices);

  free(prev.offdiag_row_indices);
  free(prev.offdiag_col_indices);
  free(prev.offdiag_values);

  free(near_null);
  return 0;
}
