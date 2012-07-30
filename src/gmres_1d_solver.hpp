// Public interface for solution of 1D Chebyshev Poisson/Helmholtz problem
#include <blitz/array.h>
#include "TArray.hpp"

int poisson_1d(blitz::Array<double,1> & resid, 
      blitz::Array<double,1> & soln, double length,
      double helmholtz, double a_top, double a_bot,
      double b_top, double b_bot);
