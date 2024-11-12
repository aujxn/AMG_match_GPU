#include "sparse.h"
#include <assert.h>

int *row_indices;
int *col_indices;

/* Function to convert COO to CSR format */
void coo_to_csr(const COOMatrix *coo, CSRMatrix *csr) {
  int nrows = coo->nrows;
  int nnz = coo->nnz;

  csr->nrows = nrows;
  csr->ncols = coo->ncols;
  csr->nnz = nnz;

  // Allocate memory
  csr->row_ptr = (int *)calloc((nrows + 1), sizeof(int));
  csr->col_indices = (int *)malloc(nnz * sizeof(int));
  csr->values = (double *)malloc(nnz * sizeof(double));

  // Step 1: Count the number of entries per row
  for (int i = 0; i < nnz; i++) {
    int row = coo->row_indices[i];
    csr->row_ptr[row + 1]++;
  }

  // Step 2: Cumulative sum to get row_ptr
  for (int i = 0; i < nrows; i++) {
    csr->row_ptr[i + 1] += csr->row_ptr[i];
  }

  // Step 3: Temporary array to hold the current position in each row
  int *current_position = (int *)malloc(nrows * sizeof(int));
  for (int i = 0; i < nrows; i++) {
    current_position[i] = csr->row_ptr[i];
  }

  // Step 4: Fill col_indices and values
  for (int i = 0; i < nnz; i++) {
    int row = coo->row_indices[i];
    int dest = current_position[row];

    csr->col_indices[dest] = coo->col_indices[i];
    csr->values[dest] = coo->values[i];

    current_position[row]++;
  }

  free(current_position);
}

/* Function to convert CSR to COO format */
void csr_to_coo(const CSRMatrix *csr, COOMatrix *coo) {
  int nrows = csr->nrows;
  int nnz = csr->nnz;

  coo->nrows = nrows;
  coo->ncols = csr->ncols;
  coo->nnz = nnz;

  // Allocate memory
  coo->row_indices = (int *)malloc(nnz * sizeof(int));
  coo->col_indices = (int *)malloc(nnz * sizeof(int));
  coo->values = (double *)malloc(nnz * sizeof(double));

  int k = 0;
  for (int row = 0; row < nrows; row++) {
    for (int idx = csr->row_ptr[row]; idx < csr->row_ptr[row + 1]; idx++) {
      coo->row_indices[k] = row;
      coo->col_indices[k] = csr->col_indices[idx];
      coo->values[k] = csr->values[idx];
      k++;
    }
  }
}

void csr_to_symmcoo(const CSRMatrix *csr, SymmCOOMatrix *coo) {

  assert(csr->nnz % 2 == 0);

  int nweights = (csr->nnz - csr->nrows) / 2;
  coo->offdiag_nnz = nweights;
  coo->dim = csr->nrows;

  int k = 0;
  for (int i = 0; i < csr->nrows; i++) {
    for (int idx = csr->row_ptr[i]; idx < csr->row_ptr[i + 1]; idx++) {
      int j = csr->col_indices[idx];
      if (j > i) {
        coo->offdiag_row_indices[k] = i;
        coo->offdiag_col_indices[k] = j;
        coo->offdiag_values[k] = csr->values[idx];
        k += 1;
      } else if (i == j) {
        coo->diag[i] = csr->values[idx];
      }
    }
  }
}

/* Sparse Matrix-Vector Multiplication using CSR format with OpenMP */
void csr_spmv(const CSRMatrix *csr, const double *x, double *y) {
  int nrows = csr->nrows;

#pragma omp parallel for
  for (int row = 0; row < nrows; row++) {
    double sum = 0.0;
    for (int idx = csr->row_ptr[row]; idx < csr->row_ptr[row + 1]; idx++) {
      int col = csr->col_indices[idx];
      double val = csr->values[idx];
      sum += val * x[col];
    }
    y[row] = sum;
  }
}

/* Sparse Matrix-Vector Multiplication using SymmCOO format */
void coo_symm_spmv(const SymmCOOMatrix *a, const double *x, double *y) {
#pragma omp parallel for
  for (int k = 0; k < a->dim; k++) {
    y[k] = a->diag[k] * x[k];
  }

  for (int k = 0; k < a->offdiag_nnz; k++) {
    int i = a->offdiag_row_indices[k];
    int j = a->offdiag_col_indices[k];
    double val = a->offdiag_values[k];
    y[i] += x[j] * val;
    y[j] += x[i] * val;
  }
}

// Function to load binary data from file
void load_binary_file(const char *filename, void *buffer, size_t size) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error opening file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  fread(buffer, size, 1, file);
  fclose(file);
}

/*
int ptap_symbolic_coo(const COOMatrix *p, const COOMatrix *a, COOMatrix *ap,
                      COOMatrix *ptap) {

  const int n_rows_p = p->nrows;
  const int n_cols_p = p->ncols;
  const int nnz_p = p->nnz;
  const int *const rowidx_p = p->row_indices;
  const int *const colidx_p = p->col_indices;
  const double *const values_p = p->values;
  const double nnz_a = a->nnz;
  const int *const rowidx_a = a->row_indices;
  const int *const colidx_a = a->col_indices;
  int *nnz_ptap = &(ptap->nnz);
  int **out_rowidx = &(ptap->row_indices);
  int **out_colidx = &(ptap->col_indices);

  // First compute A*P
  int nnz_ap;
  int *rowidx_ap;
  int *colidx_ap;

  // Allocate max possible size for AP
  nnz_ap = nnz_a * nnz_p;
  rowidx_ap = (int *)malloc(nnz_ap * sizeof(int));
  colidx_ap = (int *)malloc(nnz_ap * sizeof(int));

  // Compute AP symbolically
  int ap_count = 0;
  for (int i = 0; i < nnz_a; i++) {
    const int row_a = rowidx_a[i];
    const int col_a = colidx_a[i];

    for (int j = 0; j < nnz_p; j++) {
      if (colidx_p[j] == row_a) {
        rowidx_ap[ap_count] = rowidx_p[j];
        colidx_ap[ap_count] = col_a;
        ap_count++;
      }
    }
  }
  nnz_ap = ap_count;
  ap->row_indices = rowidx_ap;
  ap->col_indices = colidx_ap;
  ap->nnz = nnz_ap;

  // Now compute P^T * (AP)
  // Allocate max possible size for final result
  *nnz_ptap = nnz_p * nnz_ap;
  *out_rowidx = (int *)malloc(*nnz_ptap * sizeof(int));
  *out_colidx = (int *)malloc(*nnz_ptap * sizeof(int));

  // Compute P^T * AP symbolically
  int ptap_count = 0;
  for (int i = 0; i < nnz_p; i++) {
    const int row_pt = colidx_p[i];

    for (int j = 0; j < nnz_ap; j++) {
      if (rowidx_ap[j] == rowidx_p[i]) {
        (*out_rowidx)[ptap_count] = row_pt;
        (*out_colidx)[ptap_count] = colidx_ap[j];
        ptap_count++;
      }
    }
  }
  *nnz_ptap = ptap_count;

  return 0;
}

int ptap_numeric_coo(const COOMatrix *p, const COOMatrix *a,
                     const COOMatrix *ap, COOMatrix *ptap) {

  const int n_rows_p = p->nrows;
  const int n_cols_p = p->ncols;
  const int nnz_p = p->nnz;
  const int *const rowidx_p = p->row_indices;
  const int *const colidx_p = p->col_indices;
  const double *const values_p = p->values;
  const double nnz_a = a->nnz;
  const int *const rowidx_a = a->row_indices;
  const int *const colidx_a = a->col_indices;
  const double *const values_a = a->values;
  const int nnz_ptap;
  const int *const rowidx_ptap;
  const int *const colidx_ptap;
  double *const values_ptap;
  // First compute A*P
  int nnz_ap = ap->nnz;
  int *rowidx_ap = ap->row_indices;
  int *colidx_ap = ap->col_indices;
  double *values_ap;

  // Allocate max possible size for AP
  nnz_ap = nnz_a * nnz_p;
  rowidx_ap = (int *)malloc(nnz_ap * sizeof(int));
  colidx_ap = (int *)malloc(nnz_ap * sizeof(int));
  values_ap = (double *)malloc(nnz_ap * sizeof(double));

  // Compute AP numerically
  int ap_count = 0;
  for (int i = 0; i < nnz_a; i++) {
    const int row_a = rowidx_a[i];
    const int col_a = colidx_a[i];
    const double val_a = values_a[i];

    for (int j = 0; j < nnz_p; j++) {
      if (colidx_p[j] == row_a) {
        rowidx_ap[ap_count] = rowidx_p[j];
        colidx_ap[ap_count] = col_a;
        values_ap[ap_count] = val_a * values_p[j];
        ap_count++;
      }
    }
  }
  nnz_ap = ap_count;

// Now compute P^T * (AP)
#pragma omp parallel for
  for (int k = 0; k < nnz_ptap; k++) {
    values_ptap[k] = 0;
  }

  // Compute P^T * AP numerically
  for (int i = 0; i < nnz_p; i++) {
    const int row_pt = colidx_p[i];
    const double val_pt = values_p[i];

    for (int j = 0; j < nnz_ap; j++) {
      if (rowidx_ap[j] == rowidx_p[i]) {
        // Find position in final matrix
        for (int k = 0; k < nnz_ptap; k++) {
          if (rowidx_ptap[k] == row_pt && colidx_ptap[k] == colidx_ap[j]) {
            values_ptap[k] += val_pt * values_ap[j];
            break;
          }
        }
      }
    }
  }

  // Clean up temporary arrays
  free(rowidx_ap);
  free(colidx_ap);
  free(values_ap);

  return SFEM_SUCCESS;
}
*/

int compare_indices_coo(const void *a, const void *b) {
  int idx_a = *(const int *)a;
  int idx_b = *(const int *)b;

  if (row_indices[idx_a] != row_indices[idx_b])
    return row_indices[idx_a] - row_indices[idx_b];
  else
    return col_indices[idx_a] - col_indices[idx_b];
}

// TODO this could probably operate on COO matrices and be a public API in the
// sparse matrix header... But I'm reusing the cycle swap function so that also
// would need to be exported as helper
void sum_duplicates(int *rows, int *cols, double *values, int *indices,
                    int *N_ptr) {
  int N = *N_ptr;

  // Assign the global pointers
  row_indices = rows;
  col_indices = cols;

  for (int i = 0; i < N; i++) {
    indices[i] = i;
  }

  // Sort the indices array
  qsort(indices, N, sizeof(int), compare_indices_coo);
  cycle_leader_swap(rows, cols, values, indices, N);

  // Compact the arrays
  int write_pos = 0;
  int prev_row = -1;
  int prev_col = -1;

  for (int k = 0; k < N; k++) {
    int idx = indices[k];
    int i = rows[idx];
    int j = cols[idx];
    double v = values[idx];

    if (i > 0 && i == prev_row && j == prev_col) {
      // Duplicate entry found; sum the values
      values[write_pos - 1] += v;
    } else {
      // New unique entry; copy to the write position
      rows[write_pos] = i;
      cols[write_pos] = j;
      values[write_pos] = v;
      write_pos++;
    }

    prev_row = i;
    prev_col = j;
  }

  *N_ptr = write_pos;
}

void cycle_leader_swap(int *rows, int *cols, double *values, int *indices,
                       int N) {
  for (int i = 0; i < N; i++) {
    int current = i;
    while (indices[current] != i) {
      int next = indices[current];

      int temp_row = rows[current];
      rows[current] = rows[next];
      rows[next] = temp_row;

      int temp_col = cols[current];
      cols[current] = cols[next];
      cols[next] = temp_col;

      double temp_val = values[current];
      values[current] = values[next];
      values[next] = temp_val;

      indices[current] = current;
      current = next;
    }
    indices[current] = current;
  }
}
