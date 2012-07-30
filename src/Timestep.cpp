#include "Timestep.hpp"
#include <assert.h>

namespace Timestep {
   void get_coeff(const Stepped<double> & times,
                  Stepped<double> & coeffs_left,
                  Stepped<double> & coeffs_right) {
      /* By building the Lagrange interpolating polynomial, we can derive
         a mixed implicit/explicit timestepper for arbitrary (that is,
         -not- equispaced) time levels.  In current form, this function
         supports up to four time levels (times[-2]->times[1]) */

      // First, make sure that we're being consistent, array-wise
      assert(times.size() == 4);
      assert(coeffs_left.size() == 4);
      assert(coeffs_right.size() == 4);

      /* Now, mathematically we are to approximate:
            u_t(t) = u_expl(t) + u_impl(t)
         where u_expl and u_impl are terms to be stepped implicitly and
         explicitly, respectively.  It's also "nice" to require u_impl
         be evaluated at only one timelevel (t[1]), since that reduces
         the amount of data we have to keep around.  Thus, the skeleton
         looks like:
            A*u(t[-2]) + B*u(t[-1]) + C*u(t[0]) + D*u(t[1]) =
               E*u_ex(t[-2]) + F*u_ex(t[-1]) + G*u_ex(t[0]) +
               u_impl(t[1])

         Our first task is to find A/B/C/D consistently with u_impl(t[1]).
         Formally, the job is to build the Taylor series corresponding to
         u(t[-2..1]), differentiate, and cancel (4) terms.  This is annoying.

         The Taylor expansion essentially creates a polynomial in (t-t[k]).
         Another, easier way to create this same polynomial is to use an
         interpolating form: the Lagrange interpolating polynomial.  Taking
         the derivative of this (at t=t[1]) will give the A..D coefficients
         immediately. */

      // For graceful startup, determine whether we're using a reduced-order
      // formula:
      int order = 3;
      if (times[-2] == times[-1]) {
         if (times[-1] == times[0]) {
               order = 1;
         } else order = 2;
      }
      int start_idx = -2+(3-order);
      // Zero out the coeffs:
      coeffs_left[-2] = coeffs_left[-1] = coeffs_left[0] = coeffs_left[1] = 0;
      coeffs_right[-2] = coeffs_right[-1] = 
         coeffs_right[0] = coeffs_right[1] = 0;
      
      /* For reference, the form of the lagrange interpolating polynomial is:
         P(t) = f(t0) * (t-t1) * (t-t2) * ... / (t0-t1) * (t0-t2) ... +
                f(t1) * (t-t0) * (t-t2) * ... / (t1-t0) * (t1-t2) ...
                + ... */
      // Compute d/dt(interpolating polynomial)
      for (int i = start_idx; i <= 0; i++) {
         /* For all terms but the last one, there is a (t-t1) term.  After
            applying the product rule, only the sub-term in which the (t-t1)
            is removed contributes to the coefficient. */
         double numer = 1;
         double denom = times[i] - times[1];
         for (int j = start_idx; j <= 0; j++) {
            if (i==j) continue; // (t_i - t_i) term doesn't exist
            numer = numer * (times[1] - times[j]);
            denom = denom * (times[i] - times[j]);
         }
         coeffs_left[i] = numer/denom;
      }
      /* For the coeffs_left[1] term, there is no (times[1] - times[1]) term.
         So, we have to do the product rule the hard way. */
      {
         double denom = 1, numsum = 0;
         for (int i = start_idx; i <= 0; i++)
            denom = denom * (times[1] - times[i]);
         for (int skipme = start_idx; skipme <= 0; skipme++) {
            double numer = 1;
            for (int j = start_idx; j <= 0; j++) {
               if (j == skipme) continue;
               numer = numer * (times[1] - times[j]);
            }
            numsum = numsum + numer;
         }
         coeffs_left[1] = numsum/denom;
      }

      /* Now, we have the right-hand-side coefficients to deal with.
         Fortunately, we have a consistent framework -- everything is evaluated
         at t[1].  For u_t and u_impl, this is no problem.  For u_expl, we
         have to *extrapolate* using the (interpolating) polynomial.  The
         stability of this extrapolation translates into the stability of
         the timestepping scheme overall. */
      coeffs_right[1] = 1; // Coefficient for implicit term
      for (int i = start_idx; i <= 0; i++) {
         double numer = 1, denom = 1;
         for (int j = start_idx; j<= 0; j++) {
            if (i == j) continue;
            numer = numer * (times[1] - times[j]);
            denom = denom * (times[i] - times[j]);
         }
         coeffs_right[i] = numer/denom;
      }
      return;
   }
}
