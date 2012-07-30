#include "../TArray.hpp"
#include "../T_util.hpp"
#include "../Parformer.hpp"
#include "../Par_util.hpp"
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <mpi.h>

using namespace TArrayn;
using namespace Transformer;
using blitz::Array;
using blitz::firstIndex;
using blitz::secondIndex;
using blitz::thirdIndex;
using std::atoi;
using std::cout;
using std::endl;

/* Test case to check DFT, DCT, DST calculations of derivative.  The test
   function (in 3D) is an off-center Gaussian, strong enough to decay to
   essentially zero at the boundaries.  This smooth function provides a
   very nice test case, and running this program with different sizes
   (specified on the command line) demonstrates exponential convergence
   to an accurate solution.  Around size = 50, roundoff error begins to
   dominate (at the 10^-15 level). */

int main(int argc, char * argv[]) {
   MPI_Init(&argc, &argv);
   int sz = 0;
   if (argc > 1) { /* Size specified on command line */
      sz = atoi(argv[1]);
   }
   if (sz == 0) {
      sz = 64;
   }
   firstIndex ii;
   secondIndex jj;
   thirdIndex kk;
   int returnOK = 0;
   /* Arrays for differentiation and checking */
   DTArray & src = *alloc_array(sz,sz,sz),
           & dst = *alloc_array(sz,sz,sz),
           & chk = *alloc_array(sz,sz,sz);
   /* Arrays for coodinates */
   Array<double,1> xx(split_range(sz)), yy(sz), zz(sz);

   double maxsol = 0;
   double maxdiff = 0;

   { // DCT/DST derivative
      Trans1D d1(sz,sz,sz,firstDim,REAL),
              d2(sz,sz,sz,secondDim,REAL),
              d3(sz,sz,sz,thirdDim,REAL);
      xx = -M_PI/2 + M_PI*(2*ii+1.0)/(2*double(sz));
      yy = -M_PI/2 + M_PI*(2*ii+1.0)/(2*double(sz));
      zz = -M_PI/2 + M_PI*(2*ii+1.0)/(2*double(sz));

      /* Calculate source gaussian */

      src = exp(-16*((xx(ii)-.1)*(xx(ii)-.1)
               +(yy(jj)-.1)*(yy(jj)-.1)
               +(zz(kk)-.1)*(zz(kk)-.1)));


      /* Dimension 1 */
      chk = -16*2*(xx(ii)-.1)*src;
      maxsol = psmax(max(abs(chk)));
      deriv_dct(src,d1,dst);
      chk = (chk - dst)/maxsol;
      maxdiff = psmax(max(abs(chk)));
      if (maxdiff > 1e-8) returnOK = 1;
      cout << "DCT_x(" << sz << "): " << maxdiff << endl;

      
      chk = -16*2*(xx(ii)-.1)*src;
      maxsol = psmax(max(abs(chk)));
      deriv_dst(src,d1,dst);
      chk = (chk - dst)/maxsol;
      maxdiff = psmax(max(abs(chk)));
      if (maxdiff > 1e-8) returnOK = 1;
      cout << "DST_x(" << sz << "): " << maxdiff << endl;

      /* Dimension 2 */
      chk = -16*2*(yy(jj)-.1)*src;
      maxsol = psmax(max(abs(chk)));
      deriv_dct(src,d2,dst);
      chk = (chk - dst)/maxsol;
      maxdiff = psmax(max(abs(chk)));
      if (maxdiff > 1e-8) returnOK = 1;
      cout << "DCT_y(" << sz << "): " << maxdiff << endl;
      
      
      chk = -16*2*(yy(jj)-.1)*src;
      maxsol = psmax(max(abs(chk)));
      deriv_dst(src,d2,dst);
      chk = (chk - dst)/maxsol;
      maxdiff = psmax(max(abs(chk)));
      if (maxdiff > 1e-8) returnOK = 1;
      cout << "DST_y(" << sz << "): " << maxdiff << endl;

      /* Dimension 3*/
      chk = -16*2*(zz(kk)-.1)*src;
      maxsol = psmax(max(abs(chk)));
      deriv_dct(src,d3,dst);
      chk = (chk - dst)/maxsol;
      maxdiff = psmax(max(abs(chk)));
      if (maxdiff > 1e-8) returnOK = 1;
      cout << "DCT_z(" << sz << "): " << maxdiff << endl;
      
      
      chk = -16*2*(zz(kk)-.1)*src;
      maxsol = psmax(max(abs(chk)));
      deriv_dst(src,d3,dst);
      chk = (chk - dst)/maxsol;
      maxdiff = psmax(max(abs(chk)));
      if (maxdiff > 1e-8) returnOK = 1;
      cout << "DST_z(" << sz << "): " << maxdiff << endl;
   }
   if (0){ // DFT derivative
      xx = -M_PI + 2*M_PI*ii/double(sz);
      yy = -M_PI + 2*M_PI*ii/double(sz);
      zz = -M_PI + 2*M_PI*ii/double(sz);
      Trans1D d1(sz,sz,sz,firstDim,FOURIER),
              d2(sz,sz,sz,secondDim,FOURIER),
              d3(sz,sz,sz,thirdDim,FOURIER);
      /* Calculate source gaussian */
      /* The strength of the Gaussian is scaled so that the values at the
         +-PI endpoints of the domain match the values at the +-PI/2
         endpoints for the real-to-real transforms. */
      src = exp(-4*((xx(ii)-.1)*(xx(ii)-.1)
               +(yy(jj)-.1)*(yy(jj)-.1)
               +(zz(kk)-.1)*(zz(kk)-.1)));
      { // Dimension 1
         chk = -4*2*(xx(ii)-.1)*src;
         deriv_fft(src,d1,dst);
//         deriv_fft(src,tmp,dst,firstDim);
         chk = (chk - dst)/max(fabs(chk));
         if (max(fabs(chk)) > 1e-8) returnOK = 1;
         cout << "DFT_x(" << sz << "): " << max(fabs(chk)) << endl;
      }
      { // Dimension 2
         chk = -4*2*(yy(jj)-.1)*src;
         deriv_fft(src,d2,dst);
//         deriv_fft(src,tmp,dst,secondDim);
         chk = (chk - dst)/max(fabs(chk));
         if (max(fabs(chk)) > 1e-8) returnOK = 1;
         cout << "DFT_y(" << sz << "): " << max(fabs(chk)) << endl;
      }
      { // Dimension 3
         chk = -4*2*(zz(kk)-.1)*src;
         deriv_fft(src,d3,dst);
//         deriv_fft(src,tmp,dst,thirdDim);
         chk = (chk - dst)/max(fabs(chk));
         if (max(fabs(chk)) > 1e-8) returnOK = 1;
         cout << "DFT_z(" << sz << "): " << max(fabs(chk)) << endl;
      }
   }
   MPI_Finalize();
   return returnOK;
   
}
   
