#include "interpolation.h"

/*
void piecewise_constant(const int *partition, const double *near_null,
                        COOMatrix *p) {
  for (int i = 0; i < p->nrows; i++) {
    p->row_indices[i] = i;
    p->col_indices[i] = partition[i];
    p->values[i] = near_null[i];
  }
}
*/

void coarsen(const SymmCOOMatrix *a, const PiecewiseConstantTransfer *p,
             SymmCOOMatrix *a_coarse) {

  int fine_nnz = a->offdiag_nnz;

  int *row_indices = (int *)malloc(fine_nnz * sizeof(int));
  int *col_indices = (int *)malloc(fine_nnz * sizeof(int));
  double *values = (double *)malloc(fine_nnz * sizeof(double));
  double *acoarse_diag = (double *)calloc(p->coarse_dim, sizeof(double));

  int *sort_indices = (int *)malloc(fine_nnz * sizeof(int));
  int new_nnz = fine_nnz;

  pwc_restrict(p, a->diag, acoarse_diag);

  int write_pos = 0;
  for (int k = 0; k < fine_nnz; k++) {
    int i = a->offdiag_row_indices[k];
    int j = a->offdiag_col_indices[k];
    double val = a->offdiag_values[k];

    int coarse_i = p->partition[i];
    int coarse_j = p->partition[j];
    double coarse_val = p->weights[i] * val * p->weights[j];
    if (coarse_j > coarse_i) {
      row_indices[write_pos] = coarse_i;
      col_indices[write_pos] = coarse_j;
      values[write_pos] = coarse_val;
      write_pos++;
    } else if (coarse_i > coarse_j) {
      row_indices[write_pos] = coarse_j;
      col_indices[write_pos] = coarse_i;
      values[write_pos] = coarse_val;
      write_pos++;
    } else {
      acoarse_diag[coarse_i] += coarse_val;
    }
  }

  sum_duplicates(row_indices, col_indices, values, sort_indices, &new_nnz);

  a_coarse->diag = acoarse_diag;
  a_coarse->dim = p->coarse_dim;
  a_coarse->offdiag_nnz = new_nnz;
  a_coarse->offdiag_row_indices = (int *)malloc(new_nnz * sizeof(int));
  a_coarse->offdiag_col_indices = (int *)malloc(new_nnz * sizeof(int));
  a_coarse->offdiag_values = (double *)malloc(new_nnz * sizeof(double));

#pragma omp parallel for
  for (int k = 0; k < new_nnz; k++) {
    a_coarse->offdiag_row_indices[k] = row_indices[k];
    a_coarse->offdiag_col_indices[k] = col_indices[k];
    a_coarse->offdiag_values[k] = values[k];
  }

  free(values);
  free(col_indices);
  free(row_indices);
  free(sort_indices);
}

void pwc_interpolate(const PiecewiseConstantTransfer *p, const double *v_coarse,
                     double *v) {
#pragma omp parallel for
  for (int k = 0; k < p->fine_dim; k++) {
    v[k] = v_coarse[p->partition[k]] * p->weights[k];
  }
}

void pwc_restrict(const PiecewiseConstantTransfer *p, const double *v,
                  double *v_coarse) {
#pragma omp parallel for
  for (int k = 0; k < p->coarse_dim; k++) {
    v_coarse[k] = 0.0;
  }

  for (int k = 0; k < p->fine_dim; k++) {
    v_coarse[p->partition[k]] += v[k] * p->weights[k];
  }
}
