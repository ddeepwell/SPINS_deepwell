#include "../ESolver.hpp"
#include "../TArray.hpp"
#include "../T_util.hpp"
#include "../Timestep.hpp"
#include "../Transformer.hpp"
#include <blitz/array.h>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include "../Par_util.hpp"

using namespace TArray;
using namespace ESolver;
using namespace std;
using namespace Timestep;
using namespace Transformer;

using blitz::Array;
using blitz::firstIndex;
using blitz::secondIndex;
using blitz::thirdIndex;

firstIndex ii; secondIndex jj; thirdIndex kk;

/* Test case for triply periodic heat equation */


int main(int argc, char * argv[]) {
   MPI_Init(0,0);
   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   int szx, szy, szz;
   szx = szy = szz = 0;
   if (argc > 1) {
      szx = szy = szz = atoi(argv[1]);
   } if (szx == 0) {
      szx = szy = szz = 64;
   }
   /* Our final solution will be proportional to exp(-x^2/16) = exp(-(x/4)^2),
      and we want the solution at the edges to be extremely small -- O(10^-15).

      Hence, exp(-(L/4)^2) ~= 10^-15 ~= exp(-34), so L ~= 23, with the domain
      running from [-L,L] */

   double Lx, Ly, Lz;
   Lx = Ly = Lz = M_PI; //23;
   Lx = Ly = Lz = 23;

   // Shifting arrays
   Stepped<double> times(4); // times
   Stepped<double> coeffs_left(4), coeffs_right(4); // timestep coeffs
   
   Stepped<DTArray> us(4); // heat values
   us.setp(-2,alloc_array(szx, szy, szz)); // Initialize members
   us.setp(-1,alloc_array(szx, szy, szz));
   us.setp( 0,alloc_array(szx, szy, szz));
   us.setp( 1,alloc_array(szx, szy, szz));
   
   
   DTArray & rhs = *alloc_array(szx, szy, szz); // rhs for implicit solve
   DTArray & exact_soln = *alloc_array(szx, szy, szz); // exact solution

   Array<double,1> xx(split_range(szx)), yy(szy), zz(szz); // grid
   
   xx = -Lx + 2*Lx*(ii+0.5)/szx;
   yy = -Ly + 2*Ly*(ii+0.5)/szy;
   zz = -Lz + 2*Lz*(ii+0.5)/szz;
   // Elliptic solver
   TransWrapper spec_wrapper(szx,szy,szz,FOURIER,FOURIER,FOURIER);
   ElipSolver solver(0,&spec_wrapper, 2*Lx, 2*Ly, 2*Lz); 

   int num_out = 8;
   double max_dt = 1.0/(32*num_out);
   double fin_t = 1;
   double start_dt = max_dt/32;

   // Initialize
   us[1] = exp(-(xx(ii)*xx(ii)+yy(jj)*yy(jj)+zz(kk)*zz(kk))/4/(1+times[1]));

/*   double mode = 1;
   us[1] = cos(mode*xx(ii)); */
   
   int counter = 0;
   double now = MPI_Wtime();
   while (times[1] < (fin_t-1e-12)) {
//      cout << "Taking timesep, t=" << times[1] << "\n";
      double dt;
      times.shift(); us.shift(); // Shift timelevels and u values
//      times[-2] = times[-1] = times[0];
      if (times[-1] == 0) // startup -- using a lower-order method
         dt = start_dt;
      else if ( (times[0] - times[-1]) < (2*max_dt/3)) // increase timestep
         dt = 2*(times[0] - times[-1]);
      if (dt > max_dt/2.1) // hack for counter increment
         counter++;

      times[1] = times[0] + dt;
      /* Times[1] now contains our next timelevel, so get the coefficients
         for the timestepping */
      get_coeff(times, coeffs_left, coeffs_right);


      /* Base problem is:
            u_t = Lap(u), or
            dot(coeffs_left,u) = Lap(u[1]), or
            Lap(u[1]) - coeffs_left[1] * u[1] = coeffs_left[-2]*u[-2] + ...
      */
      
      rhs = coeffs_left[-2] * us[-2] +
            coeffs_left[-1] * us[-1] +
            coeffs_left[0] * us[0];
      
      solver.change_m(coeffs_left[1]);

      solver.solve(rhs,us[1],FOURIER,FOURIER,FOURIER); // Solve the incremental elliptic problem
      exact_soln = pow(1+times[1],-1.5) *
                   exp(-(xx(ii)*xx(ii)+yy(jj)*yy(jj)+zz(kk)*zz(kk))/
                        (4*(1+times[1])));
//      exact_soln = exp(-mode*mode*times[1])*cos(xx(ii)*mode);
      double max_err = psmax(max(abs(us[1]-exact_soln)))/pvmax(exact_soln);

/*      cout << times[1] << ":" <<
            us[1](szx/2,szy/2,szz/2) << " - " << 
            exact_soln(szx/2,szy/2,szz/2) <<
            ": " << us[1](szx/2,szy/2,szz/2) - 
                  exact_soln(szx/2,szy/2,szz/2) <<
            "\n"; */
     if (!myrank && counter && !(counter % num_out))
        cout << times[1] << ": " << max_err << "\n";

   }
   double later = MPI_Wtime();
   if (!myrank) fprintf(stderr,"Time taken in loop: %.2f sec\n",later-now);
   MPI_Finalize();
      
}

   
   
   

