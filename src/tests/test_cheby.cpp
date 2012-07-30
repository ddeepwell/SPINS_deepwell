#include <mpi.h>
#include <blitz/array.h>
#include "../Parformer.hpp"
#include "../Par_util.hpp"
#include "../TArray.hpp"
#include "../T_util.hpp"
#include <stdlib.h>
#include <stdio.h>


blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

using namespace blitz;

int main(int argc, char ** argv) {
   MPI_Init(&argc, &argv);
   int N = 0;
   if (argc > 1)
      N = atoi(argv[1]);
   if (N <= 0) N=32;
   TArrayn::DTArray 
      mychebtest(alloc_lbound(1,1,N), 
            alloc_extent(1,1,N), alloc_storage(1,1,N)),
      mytemp(alloc_lbound(1,1,N), 
            alloc_extent(1,1,N), alloc_storage(1,1,N));
   Transformer::Trans1D mytrans(1,1,N,TArrayn::thirdDim,Transformer::CHEBY);
   blitz::Array<double,1> xx(split_range(1)), yy(1), zz(N);
   xx = cos(M_PI*ii/(N-1));
   yy = cos(M_PI*ii/(N-1));
   zz = cos(M_PI*ii/(N-1));
   for (int p = N-1; p >= (0? N-30: 0); p--) {
      /* f */
      mychebtest = pow(zz(kk),p) + 0*(pow(yy(jj),p) + pow(zz(kk),p));
      /* Differentiate */
      deriv_cheb(mychebtest,mytrans,mytemp);
//      cout << mytemp;
      if (p)
         mychebtest = abs(p*pow(zz(kk),p-1)  - mytemp);
      else
         mychebtest = abs(mytemp);
      double maxdiff = pvmax(mychebtest);
      if (master()) {
         printf("%d: %e\n",p,maxdiff);
      }
   }
   MPI_Finalize();
   return 0;
}
   
