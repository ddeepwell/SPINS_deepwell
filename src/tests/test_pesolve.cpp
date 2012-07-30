#include "../ESolver.hpp"
#include "../TArray.hpp"
#include "../Parformer.hpp"
#include <blitz/array.h>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include "../Par_util.hpp"

using namespace TArrayn;
using namespace ESolver;
using namespace std;
using namespace Transformer;
using blitz::Array;
using blitz::firstIndex;
using blitz::secondIndex;
using blitz::thirdIndex;

/* Elliptic solver test, parallel edition.  Properly runs on more than one 
   processor */

int main(int argc, char * argv[]) {
   MPI_Init(0,0);
   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   int sz = 0; /* 3D size */
   if (argc > 1) {
      sz = atoi(argv[1]);
   }
   if (sz == 0) {
      sz = 32;
   }
   double M = 1; // Coefficient for Helmholtz problem
   double length = 2.5;
   bool returnOK = true;
   firstIndex ii; secondIndex jj; thirdIndex kk;

   DTArray & exact_soln = *alloc_array(sz,sz,sz), 
           & numer_soln = *alloc_array(sz,sz,sz), 
           & rhs        = *alloc_array(sz,sz,sz),
           & kernel     = *alloc_array(sz,sz,sz);
   TransWrapper wrapperF(sz,sz,sz,FOURIER,FOURIER,FOURIER);
   TransWrapper wrapperR(sz,sz,sz,SINE,SINE,SINE); // dual-purposed real-valued wrapper

   Array<double,1> xx(split_range(sz)), yy(sz), zz(sz);
   xx = -length/2 + length*(ii+0.5)/sz; // Grid goes from -L/2 -> L/2
   yy = -length/2 + length*(ii+0.5)/sz; // Grid goes from -L/2 -> L/2
   zz = -length/2 + length*(ii+0.5)/sz; // Grid goes from -L/2 -> L/2

   /* As test function, use a off-centre three-dimensional Gaussian pulse.  This function is
      not exactly periodic (in any sense, including possible eve/odd symmetry), so
      in the limit of exact arithmetic and infinite resolution accuracy is limited by Gibbs
      oscillations.  With the current default parameters, the accuracy limit is O(10^-11)
      (reached around 36^3 points), which is more than enough to simply test ElipSolve. */
   kernel = pow(xx(ii)-0.1,2) + pow(yy(jj)-0.1,2) + pow(zz(kk)-0.1,2); 
   exact_soln = exp(-20*kernel);
   double sumsoln = pvsum(exact_soln);
   double meansoln = sumsoln/(sz*sz*sz);
   rhs = 40*(-3 + 40*kernel)*exact_soln;
   exact_soln = exact_soln - meansoln; // subtract 0-frequency
   double sumrhs = pvsum(rhs);
   double meanrhs = sumrhs/(sz*sz*sz);
   rhs = rhs - meanrhs;

   ElipSolver imag_solve(0,&wrapperF,length,length,length);
   ElipSolver real_solve(0,&wrapperR,length,length,length);

   // Solve poisson problems
   double maxdiff;
   imag_solve.solve(rhs,numer_soln,FOURIER,FOURIER,FOURIER);
   maxdiff = psmax(max(abs(numer_soln - exact_soln)));
   if (maxdiff > 1e-8) returnOK = false;
   if (!myrank) cout << "Poisson FFT(" << sz << "): " << maxdiff << "\n";
   real_solve.solve(rhs,numer_soln,COSINE,COSINE,COSINE);
   maxdiff = psmax(max(abs(numer_soln - exact_soln)));
   if (maxdiff > 1e-8) returnOK = false;
   if (!myrank) cout << "Poisson DCT(" << sz << "): " << maxdiff << "\n";
   // Sine expansions already don't have a 0-mode, but they "must" have 0 boundaries.
   // To fake this, subtract the minimum (making the minimum value 0)
   exact_soln = exact_soln - pvmin(exact_soln);
   {
      double topleft;
      if (!myrank) topleft = rhs(0,0,0);
      MPI_Bcast(&topleft,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      rhs = rhs - topleft;
   }
   real_solve.solve(rhs,numer_soln,SINE,SINE,SINE);
   maxdiff = psmax(max(abs(numer_soln - exact_soln)));
   if (maxdiff > 1e-8) returnOK = false;
   if (!myrank) cout << "Poisson DST(" << sz << "): " << maxdiff << "\n";

   // Solve Helmholtz problems
   imag_solve.change_m(M);
   real_solve.change_m(M);

   // Re-normalize for Fourier/cosine expansions to remove 0-frequency
   exact_soln = exact_soln - pvsum(exact_soln)/(sz*sz*sz);
   rhs = rhs - pvsum(rhs)/(sz*sz*sz) - M*exact_soln;

   imag_solve.solve(rhs,numer_soln,FOURIER,FOURIER,FOURIER);
   maxdiff = psmax(max(abs(numer_soln - exact_soln)));
   if (maxdiff > 1e-8) returnOK = false;
   if (!myrank) cout << "Helmholtz FFT(" << sz << "): " << maxdiff << "\n";
   real_solve.solve(rhs,numer_soln,COSINE,COSINE,COSINE);
   maxdiff = psmax(max(abs(numer_soln - exact_soln)));
   if (maxdiff > 1e-8) returnOK = false;
   if (!myrank) cout << "Helmholtz DCT(" << sz << "): " << maxdiff << "\n";

   // Sine expansions already don't have a 0-mode, but they "must" have 0 boundaries.
   // To fake this, subtract the minimum (making the minimum value 0)
   exact_soln = exact_soln - pvmin(exact_soln);
   {
      double topleft;
      if (!myrank) topleft = rhs(0,0,0);
      MPI_Bcast(&topleft,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      rhs = rhs - topleft;
   }
   real_solve.solve(rhs,numer_soln,SINE,SINE,SINE);
   maxdiff = psmax(max(abs(numer_soln - exact_soln)));
   if (maxdiff > 1e-8) returnOK = false;
   if (!myrank) cout << "Helmholtz DST(" << sz << "): " << maxdiff << "\n";
   MPI_Finalize();
   return (returnOK? 0:1);
}
