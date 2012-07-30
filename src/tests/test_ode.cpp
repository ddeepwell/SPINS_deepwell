#include "../Timestep.hpp"
#include <cstdio>
#include <iostream>
#include <cmath>

using namespace std;
using namespace Timestep;

/* Test case for an ODE solver

   DE to solve is

   y' = -2y + sin(t)

   Exact solution is y = (y0+1/5) exp(-2t) + 2/5 sin(t) - 1/5 cos(t)

*/

int main(int argc, char * argv[]) {
   int numouts = 0;
   if (argc > 1) {
      numouts = atoi(argv[1]);
   } if (!numouts)
      numouts = 32;
   double dt = (1.0/128)/numouts;
   double start_dt = dt / 16;
   Stepped<double> times(4), ys(4), forcing(4);
   Stepped<double> lhs_coef(4), rhs_coef(4);

   double fin = 1, y0 = 1;

   double exact, rhs;

   ys[1] = y0;
   int counter = 0;
   while (times[1] < fin-1e-12) {
      counter++;
      times.shift();
      ys.shift();
      forcing.shift();
      if (times[-1] == times[-2]) {
         times[1] = times[0] + start_dt;
      } else
         times[1] = times[0] + min(dt,2*(times[0]-times[-1]));
      get_coeff(times, lhs_coef, rhs_coef);
      forcing[0] = sin(times[0]);

      rhs = -(lhs_coef[0]*ys[0] + lhs_coef[-1]*ys[-1] + lhs_coef[-2]*ys[-2]) +
            (rhs_coef[0]*forcing[0] + rhs_coef[-1]*forcing[-1] +
             rhs_coef[-2]*forcing[-2]);
      
      ys[1] = rhs/(lhs_coef[1]+2); 
      exact = 1.2*exp(-2*times[1])+0.4*sin(times[1])-0.2*cos(times[1]);
      if (!(counter % numouts))
      cout << times[1] << ":" << ys[1] << " - " << exact << "=" << ys[1]-exact
         << "\n";
   }
   return 0;
}
      
