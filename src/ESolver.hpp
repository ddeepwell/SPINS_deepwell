/* ESolver.hpp -- header file for ESolver class and any associated functions.*/
#ifndef ESOLVER_HPP
#define ESOLVER_HPP 1

/* ESolver is essentially the "guts" of SPINS, designed to transparently solve
   problems of the form:
      f(x,y,z)*Lap(u) - G*u = h(x,y,z)

   These problems arise (twice!) at each timestep in the discretization of the 
   Navier-Stokes equations.  Firstly, under a splitting method pressure is
   calculated as
      Lap(pressure) = div(u,v,w),
   where (u,v,w) are pressure-free projected velocities.  Without interaction
   with boundaries, pressure acts (through its gradient) to make the velocities
   divergence-free.  In this case, clearly the constant G (above) is 0, making
   this a Poisson problem.

   Secondly, with implicit diffusion, the momentum problem becomes:
      mu(x,y,z)*Lap(u) - u/dt = f(x,y,z),
   a Helmholtz problem where mu(x,y,z) is the viscosity.  If there is
   additionally a sponge layer, the diffusion may be anisotropic.  This
   is not conceptually different, but requires an equation of the form:
      muf(x,y,z)*uxx + mug(x,y,z)*uyy + muh(x,y,z)*uzz - u/dt = f(x,y,z).

   In the general case, ESolver needs to deal with a volume split over
   multiple processors and solutions that are not separable -- it must deal
   with the 3D domain as a single entity.  The only way to do this is to build
   a full finite-difference preconditioner (to be written, using multigrid)
   for the operator and solve with an iterative scheme (GMRES). 
   
   Separable dimensions are those that reduce the differential operator to
   an algebraic one under a Fourier Transform (DFT/DCT/DST).  The resulting
   problem is then a repeated one in 1 or 2 dimensions, which is significantly
   easier to solve.  Indeed, if the arallelized dimension is separable, then
   the problem of parallel multigrid can be avoided entirely.

   This suggests that ESolver breaks up into a couple different cases:
   1) Triply-separable with constant Jacobian, so take the spectral transform
      and divide by k^2+l^2+m^2
   2) Doubly-separable, with (possibly mapped) Chebyshev vertical expansion.
      In this case, we have only one dimension at a time to consider, so take
      the spectral transform on x/y and solve the tridiagonal FD problem with
      k^2+l^2 on the diagonal
   3) Singly-separable (spectral along y), requires a 2D multigrid problem.
   4) None-separable, requires a 3D multigrid problem.  For now we won't touch
      this case, since writing 3D multigrid will be a bit finicky (although not
       mathematically harder)
*/
#include "TArray.hpp"
#include "Splits.hpp"
#include "Parformer.hpp"
#include <blitz/array.h>
#include "grad.hpp"

namespace ESolver {
   using TArrayn::DTArray;
   using TArrayn::CTArray;
   using namespace Transformer;
   
   class ElipSolver {
      /* ESolver 0.1 -- triply-periodic FFT, solving:
         Lap(u) - M*u = f(x,y,z) */
      /* ElipSolver 0.2 -- adding sin/cos expansions */
      
      public:
         /* The mechanics of real/spectral transformation is handled by the
            TransWrapper class.  This means that ElipSolver doesn't have to
            directly handle temporary arrays, nor does it have to calculate
            wavenumbers (and normalization factors) by itself. */
         /* Note that the gradient operator is now specified.  We can't really
            leverage the derivatives in it since we don't -take- derivatives
            over the entire field, but we can use the jacobian info inside
            for our calculations.  (a 3D multigrid -could- use grad directly,
            but we're not writing that now.) */
         ElipSolver(double M,TransWrapper * spec_space, TArrayn::Grad * in_grad);
         ~ElipSolver(); // Destructor
         /* Solve the Helmholtz/Poisson problem with the given rhs, storage
            in output. */
         void solve(DTArray & rhs, DTArray & output,
               S_EXP type_x=FOURIER, S_EXP type_y=FOURIER, 
               S_EXP type_z=FOURIER, double zbc_a=0, double zbc_b = 0,
               double xbc_a=0, double xbc_b=0,
               double ybc_a=0, double ybc_b=0);
         /* For timestepping: reset the Helmholtz parameter */
         void change_m(double M);
      private:
         double M; // Helmholtz parameter 
         TArrayn::Grad * gradient;
         S_EXP t_types[3]; // Transform types

         /* Specialization of solve to the case when all three
            dimensions are non-mapped trig expansions.  The
            derivatives become algebraic after transform, meaning
            no GMRES required */
         void threespec_solve(DTArray & rhs, DTArray & output,
               S_EXP type_x, S_EXP type_y, S_EXP type_z);

         /* Specialization for one Chebyshev dimension (vertical),
            including (identical) BC types for the top and bottom */
         void chebz_solve(DTArray & rhs, DTArray & output,
               S_EXP type_x, S_EXP type_y, double zbc_a, double zbc_b);

         /* Specialization for the 2D GMRES/Multigrid solve,
            with identical BCs at all solid interfaces */
         void twodeemg_solve(DTArray & rhs, DTArray & output,
               S_EXP type_x, S_EXP type_y, double zbc_a, double zbc_b, double xbc_a, double xbc_b);

         TransWrapper * spec_transform;

         // Data members for the 1D (z) Chebyshev case, which requires
         // a parallel transpose for data ordering
         Transposer<complex<double> > * cTransposer;
         Transposer<double> * rTransposer;
         DTArray * rtransyz;
         CTArray * ctransyz;

         // Data members for the 2D multigrid solve, which operates on a
         // Nx1xM 3D array
         DTArray * twod_u, * twod_f;
         

   };

}  // End namespace


#endif // ESOLVER_HPP
