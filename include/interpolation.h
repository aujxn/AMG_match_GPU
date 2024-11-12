#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "sparse.h"

typedef struct {
  int fine_dim;
  int coarse_dim;
  // Length of `fine_dim` and values indicate coarse grid indices
  int *partition;
  // Length of `fine_dim` and values weight the PWC gridfunction
  double *weights;
} PiecewiseConstantTransfer;

// Internally allocates a workspace with same memory requirement as `a`
// (this could be passed in as arg...)
void coarsen(const SymmCOOMatrix *a, const PiecewiseConstantTransfer *p,
             SymmCOOMatrix *a_coarse);

void pwc_interpolate(const PiecewiseConstantTransfer *p, const double *v_coarse,
                     double *v);
void pwc_restrict(const PiecewiseConstantTransfer *p, const double *v,
                  double *v_coarse);

#endif // INTERPOLATION_H
