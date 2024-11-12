#ifndef SPARSE_H
#define SPARSE_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// COO matrix struct
typedef struct {
  int nrows;        // Number of rows
  int ncols;        // Number of columns
  int nnz;          // Number of non-zero entries
  int *row_indices; // Row indices of non-zero entries
  int *col_indices; // Column indices of non-zero entries
  double *values;   // Non-zero values
} COOMatrix;

typedef struct {
  int dim; // Number of rows / columns

  int offdiag_nnz; // Length of 3 below arrays, or half the actualy number of
                   // off diagonal non-zero entries
  int *offdiag_row_indices; // Row indices of off diagonal non-zero entries
  int *offdiag_col_indices; // Column indices of off diagonal non-zero entries
  double *offdiag_values;   // Off diagonal non-zero values

  double *diag; // Diagonal values
} SymmCOOMatrix;

// CSR matrix struct
typedef struct {
  int nrows;        // Number of rows
  int ncols;        // Number of columns
  int nnz;          // Number of non-zero entries
  int *row_ptr;     // Row pointers
  int *col_indices; // Column indices
  double *values;   // Non-zero values
} CSRMatrix;

/* Function declarations */
void coo_to_csr(const COOMatrix *coo,
                CSRMatrix *csr); // TODO handle duplicates and non-sorted...
void csr_to_coo(const CSRMatrix *csr, COOMatrix *coo);
void csr_to_symmcoo(const CSRMatrix *csr, SymmCOOMatrix *coo);
void csr_spmv(const CSRMatrix *csr, const double *x, double *y);
// void csr_spmm(const CSRMatrix *a, const CSRMatrix *b, double *y);
void coo_symm_spmv(const SymmCOOMatrix *a, const double *x, double *y);
void load_binary_file(const char *filename, void *buffer, size_t size);

/* Helper Functions */
void sum_duplicates(int *rows, int *cols, double *values, int *indices,
                    int *N_ptr);
void cycle_leader_swap(int *rows, int *cols, double *values, int *indices,
                       int N);
#endif // SPARSE_H
