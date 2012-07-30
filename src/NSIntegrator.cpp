#include "NSIntegrator.hpp"
#include "BaseCase.hpp"

/* And default global values for filtering */
double f_cutoff = 0.6, f_order = 2, f_strength = 20;

// Hack -- if a density tracer is really a perturbation density, it should
// have zero boundary conditions on a cosine/free-slip grid.  Otherwise,
// the zero-derivative BC is more appropriate.  Short of changing the function
// interface to specify this per-tracer, use a single global boolean
bool zero_tracer_boundary = false; 

namespace NSIntegrator{ 
   S_EXP get_trans_type(DIMTYPE dtype, Sym stype) {
      switch (dtype) {
         case PERIODIC:
            return FOURIER;
         case FREE_SLIP:
            if (stype == EVEN) return COSINE;
            if (stype == ODD) return SINE;
            abort(); break;
         case NO_SLIP:
            return CHEBY;
         default:
            abort(); return NONE;
      }
   }
}

template class FluidEvolve<BaseCase>; // Explicit template instantiation
