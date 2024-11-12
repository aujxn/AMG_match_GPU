#ifndef MG_BUILDER_H
#define MG_BUILDER_H

#include "interpolation.h"
#include "sparse.h"

typedef struct {
  int levels;
  double coarsening_factor;
  PiecewiseConstantTransfer **transer_operators;
  SymmCOOMatrix **matrices;
} PWCHierarchy;

int builder(const double coarsening_factor, SymmCOOMatrix *fine_mat,
            PWCHierarchy *hierarchy);

#endif // MG_BUILDER_H
