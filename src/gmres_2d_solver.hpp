#include <blitz/array.h>
#include "TArray.hpp"
#include "grad.hpp"
#include "Parformer.hpp"

/* Public interface of the 2D Poisson solver, simplified to assume uniform
   boundary conditions at all 4 (2 in a symmetric case) walls */
int poisson_2d(TArrayn::DTArray & resid, TArrayn::DTArray & soln, 
      TArrayn::Grad * gradient, Transformer::S_EXP type_x,
      double helmholtz, double zbc_a, double zbc_b, 
      double xbc_a = 0, double xbc_b = 0);
