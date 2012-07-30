#ifndef TIMESTEP_HPP
#define TIMESTEP_HPP 1
#include <vector>
#include <ostream>
#include "TArray.hpp"
using TArrayn::DTArray;
/* Timestep.hpp -- helper functions and classes to support timestepping

   With elliptic solving being done in another module (ESolver), what remains
   is to build a neat framework to support the machinery of timestepping.
   
   SPINS will use a mixed implicit/explicit timestepper.  Diffusion, which
   in the limit of high resolution creates the steepest timestep restriction,
   is handled implicitly.  (Nonlinear) advection and other forcing terms are
   stepped explicitly.  Pressure is special (in the "headache" sense) and is
   distinct.

   The basic timestepping formula is a modification of backwards
   differentiation.  In BDF, the time derivative is considered at the n+1'st
   timelevel (t[1]), so in a continuous sense:
   d/dt (f(t[1])) = stuff(f(t[1]))

   The simplicity of this method is in the use of interpolating polynomials.
   By considering f(t) to actually be a polynomial over the previous few
   timelevels (t[-3] -> t[1]), we can construct a Lagrange interpolant.
   The /derivative/ of this provides coefficients for the left hand side.
   The error of this is (dt^(number of terms)), reduced by one since the
   time derivative is integrated out.  (I.E. f(t[1]) - f(t[0]) is actually
   a first-order method.)

   Explicit terms are /extrapolated/ to the t[1] timelevel, using a Lagrange
   interpolation that does not include that time. */


/* Highly useful here is the t[1] notation.  Outside of LaTeX, where t_{n+1}
   looks pretty but is a pain to type over and over, there's rarely any
   good framework for timestepped variables.  This introduces one. */
#pragma warning (disable: 981)
namespace Timestep {
   template<class T> class Stepped {
      /* A small modification to an array -- instead of going from 0->n-1,
         this class allows indexing from -(n-2)->+1.  This allows for more
         natural semantics, like:
            tarrays[1] = dt*(tarrays[0] + rhs)
         for a prototypical Forward Euler method.  Also included is a shift
         method, to rotate the contents like a circular array.
       */
      private:
         int _n; // number contained
         std::vector<T> contents; // Contained objects
         Stepped(const Stepped<T> & rhs); // No copying
      public:
         Stepped(int n): _n(n), contents(n){ // constructor
            for (unsigned int i = 0; i < contents.size(); i++)
               contents[i] = 0;
          }
         ~Stepped() {}; // Destructor
         T& operator[](int n) { // Index operator
            return contents[n+_n-2];
         }
         const T& operator[](int n) const { // Index operator
            return contents[n+_n-2];
         }
         void shift() {  // Shift contents
            T temp = contents[0];
            for (int i = 0; i < _n-1; i++) {
               contents[i] = contents[i+1];
            }
            contents[_n-1] = temp;
         }
         int size() const { return _n;}
         template<class U> friend
            std::ostream& operator<< (std::ostream& s, const Stepped<U> &z);
   };

   template<class U>
      std::ostream & operator<< (std::ostream& s, const Stepped<U>& z);
   template<class U>
      std::ostream & operator<< (std::ostream& s, const Stepped<U>& z)  {
         for (int i = 0; i < z._n; i++) {
            s << i-z._n+2 << ": " << z.contents[i] << "\n";
         }
         return s;
      } 
   template <> class Stepped<DTArray> {
      /* Handling DTArrays by value is an absurd idea.  It's far better to deal
         with them by pointer.  This specialization of Stepped wraps DTArray*,
         and also "owns" the contained arrays for deletion. */
      private:
         int _n;
         std::vector<DTArray *> contents;
         Stepped(const Stepped<DTArray> & rhs); // no coying
      public:
         Stepped(int n): _n(n), contents(n){
            for (int i = 0; i < n; i++)
               contents[i] = 0;
          }
         ~Stepped() {
            for (int i = 0; i < _n; i++) {
               if (contents[i]) {
                  delete contents[i];
                  contents[i] = 0;
               }
            }
         }
         DTArray& operator[](int n) { // Index operator
            return *contents[n+_n-2];
         };
         const DTArray& operator[](int n) const { // Index operator
            return *contents[n+_n-2];
         };
         void shift() {  // Shift contents
            DTArray * temp = contents[0];
            for (int i = 0; i < _n-1; i++) {
               contents[i] = contents[i+1];
            }
            contents[_n-1] = temp;
         }
         int size() const { // size
            return _n;
         }
         void setp(int n, DTArray * rhs) { // Set contained pointer
            contents[n+_n-2] = rhs;
         }
         DTArray * getp(int n) const { // Get contained pointer directly
            return contents[n+_n-2];
         }
   };

   /* Gets the required coefficients for backwards differentiation and
      extrapolation, for timestepping of the form:
         u_t = u_expl(t) + u_impl(t) */
   void get_coeff(const Stepped<double> & times, 
                  Stepped<double> & coeffs_left,
                  Stepped<double> & coeffs_right);
}

#endif // TIMESTEP_HPP
