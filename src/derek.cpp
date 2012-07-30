// Required header files
#include "ESolver.hpp" // Elliptic solver
#include "TArray.hpp" // Transforming (FFT) Array
#include "T_util.hpp" // Helper functions for arrays, like output
#include "Parformer.hpp" // Parallel Transformer, for managing transpositions
#include <blitz/array.h> // Blitz array library
#include "Par_util.hpp" // Parallel utility functions
#include "grad.hpp" // Gradient operator, inclusive of any mapping

#include <iostream>
#include <math.h>
#include <mpi.h>

// Namespaces
using namespace TArrayn;
using namespace ESolver;
using namespace std;
using namespace Transformer;
using blitz::Array;

// Index placeholder variables

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;


int main(int argc, char ** argv) {
   MPI_Init(&argc, &argv);

   // The objective here is to set up and solve a Helmholtz problem
   // of the sort (grad^2 - H)*u = f
   // with appropriate (Dirichlet/Neumann) boundary conditions

   // Define problem domain
   double Lx = 1, Ly = 1, Lz = 1; // lengths
   int Nx = 128, Ny = 1, Nz = 128; // Grid sizes

   // Create and set up the Gradient operator.  
   Grad gradop(Nx, Ny, Nz,  // Grid size
         CHEBY, FOURIER, CHEBY // Expansion types
         );

   // Initialize the Jacobian matrix for the gradient.
   // Since this is an unmapped grid, the Jacobian is strictly
   // diagonal and constant-coefficient, so it can be set
   // directly

   // The value is set by d(alpha)/dx, with the computational
   // coordinate normalized to its "default" grid (1->-1 for
   // Chebyshev, -pi->pi for Fourier Periodic, and 0->pi for
   // Fourier Sine/Cosine)
   gradop.set_jac(firstDim,firstDim,2/Lx);
   gradop.set_jac(secondDim,secondDim,1/Ly);
   gradop.set_jac(thirdDim,thirdDim,2/Lz);


   // Create an object to transform from physical to spectral
   // spaces, accounting for array transposes when run in parallel

   TransWrapper spec_wrap(Nx,Ny,Nz, // grid sizes
         CHEBY,FOURIER,CHEBY // transform types
         );

   // Initialize the elliptic solver

   ElipSolver helmholtz_solve(0, // Value of helmholtz parameter, changeable
         & spec_wrap, // Transform wrapper
         & gradop // gradient operator
         );

   /* Create the grid */

   /* The x-grid (first dimension) is split amongst processors.  The
      helper function split_range(int) returns an appropriate Blitz
      Range object for the split, depending on current processor */
   Array<double,1> xg(split_range(Nx)), zg(Nz);

   // Use indical notation to assign these 1D arrays
   xg = (Lx/2)*cos(ii*M_PI/(Nx-1));
   zg = (Lz/2)*cos(ii*M_PI/(Nz-1));
   
   /* Create arrays for RHS and solution, using the parallel helper
      functions */
   DTArray rhs(alloc_lbound(Nx,Ny,Nz), // local lower bound
               alloc_extent(Nx,Ny,Nz), // local extents 
               alloc_storage(Nx,Ny,Nz)), // storage order
           soln(alloc_lbound(Nx,Ny,Nz),
                 alloc_extent(Nx,Ny,Nz),
                 alloc_storage(Nx,Ny,Nz));

   /* Initialize the right-hand-side to something appropriate */

   /* Let's solve (grad^2 - 1)*u = sin(pi*x/(Lx/2))*sin(2*pi*z/(Lz/2)),
      with Dirichlet boundary conditions (0) */

   rhs = sin(M_PI*xg(ii)/(Lx/2))*sin(2*M_PI*zg(kk)/(Lz/2));

   // Now, BCs are indeed zero, but let's reinforce that

   blitz::Range all = blitz::Range::all(); // range for "all"
   rhs(all,all,0) = 0; // top boundary (z=Lz/2)
   rhs(all,all,Nz-1) = 0; // bottom boundary
   // Left/right are more complicated because of the splitting over processors; 
   // first, we need to check if we actually -own- the left or right.

   if (rhs.lbound(firstDim) == 0) { // if 0 is the lower bound
      rhs(0,all,all) = 0; // set the right (x=Lx/2) boundary to 0
   }
   if (rhs.ubound(firstDim) == Nx-1) { // if Nx-1 is the upper bound
      rhs(Nx-1,all,all) = 0; // set the left boundary to 0
   }

   /* Now, call the elliptic solver to solve the problem */

   helmholtz_solve.change_m(1); // Change the helmholtz parameter to match
   helmholtz_solve.solve(rhs,soln, // rhs and solution arrays
         CHEBY,FOURIER,CHEBY, // expansion types
         1,0); // BCs, of the form a*u + b*normal dot grad u = <stuff>

   /* Write the RHS and solution arrays to disk */
   write_array(rhs,"rhs_field");
   write_reader(rhs,"rhs_field"); // write a .m file to read it in MATLAB

   write_array(soln,"soln_field"); // see T_util.hpp for how to write out
   write_reader(soln,"soln_field");// sequenced outputs






   MPI_Finalize(); // Close out MPI
   return 0;
}


