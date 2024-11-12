#include "interpolation.h"
#include "mg_builder.h"
#include "sparse.h"
#include <stdlib.h>

int csr_to_coo_to_csr_test();
int large_spmv_test();
int partition_test();

int main() {

  int failed =
      (csr_to_coo_to_csr_test() | large_spmv_test() | partition_test());

  if (failed) {
    printf("Some test(s) failed....\n");
  } else {
    printf("All tests successful!\n");
  }

  return failed;
}

int csr_to_coo_to_csr_test() {
  // Define COO matrix
  COOMatrix coo;
  coo.nrows = 4;
  coo.ncols = 4;
  coo.nnz = 5;

  // Allocate memory
  coo.row_indices = (int *)malloc(coo.nnz * sizeof(int));
  coo.col_indices = (int *)malloc(coo.nnz * sizeof(int));
  coo.values = (double *)malloc(coo.nnz * sizeof(double));

  // Initialize COO matrix data
  // Non-zero elements at positions:
  // (0,0) = 10
  // (1,1) = 20
  // (2,2) = 30
  // (3,0) = 40
  // (3,3) = 50

  coo.row_indices[0] = 0;
  coo.col_indices[0] = 0;
  coo.values[0] = 10.0;

  coo.row_indices[1] = 1;
  coo.col_indices[1] = 1;
  coo.values[1] = 20.0;

  coo.row_indices[2] = 2;
  coo.col_indices[2] = 2;
  coo.values[2] = 30.0;

  coo.row_indices[3] = 3;
  coo.col_indices[3] = 0;
  coo.values[3] = 40.0;

  coo.row_indices[4] = 3;
  coo.col_indices[4] = 3;
  coo.values[4] = 50.0;

  // Convert COO to CSR
  CSRMatrix csr;
  coo_to_csr(&coo, &csr);

  // Create input vector x
  double x[4] = {1.0, 1.0, 1.0, 1.0};
  // Output vector y
  double y[4] = {0.0, 0.0, 0.0, 0.0};

  // Perform SpMV
  csr_spmv(&csr, x, y);

  // Print result
  printf("Result of SpMV y = A * x:\n");
  for (int i = 0; i < csr.nrows; i++) {
    printf("y[%d] = %f\n", i, y[i]);
  }

  // Convert CSR back to COO
  COOMatrix coo2;
  csr_to_coo(&csr, &coo2);

  // Verify that coo2 matches the original coo
  int mismatch = 0;
  for (int i = 0; i < coo.nnz; i++) {
    if (coo.row_indices[i] != coo2.row_indices[i] ||
        coo.col_indices[i] != coo2.col_indices[i] ||
        coo.values[i] != coo2.values[i]) {
      mismatch = 1;
      break;
    }
  }

  if (mismatch) {
    printf("Error: COO matrices do not match!\n");
  } else {
    printf("Success: COO matrices match after conversion!\n");
  }

  // Free allocated memory
  free(coo.row_indices);
  free(coo.col_indices);
  free(coo.values);

  free(csr.row_ptr);
  free(csr.col_indices);
  free(csr.values);

  free(coo2.row_indices);
  free(coo2.col_indices);
  free(coo2.values);

  return mismatch;
}

int large_spmv_test() {
  CSRMatrix csr;

  csr.nrows = 86941;
  csr.ncols = 86941;
  csr.nnz = 1246037;

  csr.row_ptr = (int *)malloc((csr.nrows + 1) * sizeof(int));
  csr.col_indices = (int *)malloc(csr.nnz * sizeof(int));
  csr.values = (double *)malloc(csr.nnz * sizeof(double));

  double *x = (double *)malloc(csr.nrows * sizeof(double));
  double *b = (double *)malloc(csr.nrows * sizeof(double));
  double *b_verify = (double *)malloc(csr.nrows * sizeof(double));

  for (int i = 0; i < csr.nrows; i++) {
    x[i] = 1.0;
  }

  // Load CSR data from files
  load_binary_file("../data/indptr.raw", csr.row_ptr,
                   (csr.nrows + 1) * sizeof(int));
  load_binary_file("../data/indices.raw", csr.col_indices,
                   csr.nnz * sizeof(int));
  load_binary_file("../data/values.raw", csr.values, csr.nnz * sizeof(double));
  load_binary_file("../data/b.raw", b_verify, csr.nnz * sizeof(double));

  csr_spmv(&csr, x, b);

  int mismatch = 0;
  for (int i = 0; i < csr.nrows; i++) {
    if (b[i] != b_verify[i]) {
      printf("Error: spmv result incorrect! b[%d] = %f, b_verify[%d] = %f\n", i,
             b[i], i, b_verify[i]);
      mismatch = 1;
    }
  }

  if (mismatch) {
    printf("large spmv test failed, result wasn't as expected\n");
  } else {
    printf("large spmv test passed\n");
  }

  free(csr.row_ptr);
  free(csr.col_indices);
  free(csr.values);
  free(x);
  free(b);
  free(b_verify);

  return mismatch;
}

int partition_test() {
  CSRMatrix csr;
  SymmCOOMatrix *symm_coo = (SymmCOOMatrix *)malloc(sizeof(SymmCOOMatrix));
  PWCHierarchy hierarchy;
  int test_result = 0;
  double coarsening_factor = 2.0;
  csr.nrows = 1492;
  csr.ncols = 1492;
  csr.nnz = 18794;

  csr.row_ptr = (int *)malloc((csr.nrows + 1) * sizeof(int));
  csr.col_indices = (int *)malloc(csr.nnz * sizeof(int));
  csr.values = (double *)malloc(csr.nnz * sizeof(double));

  int nweights = (csr.nnz - csr.nrows) / 2;
  symm_coo->offdiag_nnz = nweights;
  symm_coo->offdiag_row_indices = (int *)malloc(nweights * sizeof(int));
  symm_coo->offdiag_col_indices = (int *)malloc(nweights * sizeof(int));
  symm_coo->offdiag_values = (double *)malloc(nweights * sizeof(double));
  symm_coo->dim = csr.nrows;
  symm_coo->diag = (double *)malloc(csr.nrows * sizeof(double));

  load_binary_file("../data/cylinder/laplace_indptr.raw", csr.row_ptr,
                   (csr.nrows + 1) * sizeof(int));
  load_binary_file("../data/cylinder/laplace_indices.raw", csr.col_indices,
                   csr.nnz * sizeof(int));
  load_binary_file("../data/cylinder/laplace_values.raw", csr.values,
                   csr.nnz * sizeof(double));

  csr_to_symmcoo(&csr, symm_coo);
  test_result = builder(coarsening_factor, symm_coo, &hierarchy);

  /*
  double prev_weight = 9999999999999999.0;
  for (int k = 0; k < 10; k++) {
    if (ws.weights[k] > prev_weight) {
      printf("FAILED weights are not sorted...");
      test_result = 1;
      break;
    }
    prev_weight = ws.weights[k];
    printf("ptr_i[%d] = %d, ptr_j[%d] = %d, weights[%d] = %.2f\n", k,
           ws.ptr_i[k], k, ws.ptr_j[k], k, ws.weights[k]);
  }
  */

  free(csr.row_ptr);
  free(csr.col_indices);
  free(csr.values);

  for (int level = 0; level < hierarchy.levels; level++) {
    SymmCOOMatrix *mat = hierarchy.matrices[level];
    free(mat->offdiag_row_indices);
    free(mat->offdiag_col_indices);
    free(mat->offdiag_values);
    free(mat->diag);
    free(mat);

    if (level < (hierarchy.levels - 1)) {
      PiecewiseConstantTransfer *p = hierarchy.transer_operators[level];

      free(p->weights);
      free(p->partition);
      free(p);
    }
  }

  free(hierarchy.transer_operators);
  free(hierarchy.matrices);

  return test_result;
}
