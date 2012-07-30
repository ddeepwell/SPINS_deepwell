/* WARNING: Science Content!

   Collection of various analysis routines that are general enough to be useful
   over more than one project */

#ifndef SCIENCE_HPP
#define SCIENCE_HPP 1

#include <blitz/array.h>
#include "TArray.hpp"
#include "NSIntegrator.hpp"



// Marek's Overturning Diagnostic
blitz::Array<double,3> overturning_2d(blitz::Array<double,3> const & rho, 
      blitz::Array<double,1> const & zgrid, TArrayn::Dimension reduce = TArrayn::thirdDim );

// Read in a 2D file and interpret it as a 2D slice of a 3D array, for
// initialization with read-in-data from a program like MATLAB
void read_2d_slice(blitz::Array<double,3> & fillme, const char * filename, 
                  int Nx, int Ny);

/* Compute vorticity */
void vorticity(TArrayn::DTArray & u, TArrayn::DTArray & v, 
      TArrayn::DTArray & w, TArrayn::DTArray * & w_x, TArrayn::DTArray * & w_y,
      TArrayn::DTArray * & w_z, double Lx, double Ly, double Lz,
      int szx, int szy, int szz,
      NSIntegrator::DIMTYPE DIM_X, NSIntegrator::DIMTYPE DIM_Y, 
      NSIntegrator::DIMTYPE DIM_Z);


// Quadrature weights
void compute_quadweights(int szx, int szy, int szz, 
      double Lx, double Ly, double Lz,
      NSIntegrator::DIMTYPE DIM_X, NSIntegrator::DIMTYPE DIM_Y,
      NSIntegrator::DIMTYPE DIM_Z);
const blitz::Array<double,1> * get_quad_x();
const blitz::Array<double,1> * get_quad_y();
const blitz::Array<double,1> * get_quad_z();

// Equation of state for seawater, polynomial fit from
// Brydon, Sun, Bleck (1999) (JGR)

inline double eqn_of_state(double T, double S){
   // Returns the density anomaly (kg/m^3) for water at the given
   // temperature T (degrees celsius) and salinity S (PSU?)

   // Constants are from table 4 of the above paper, at pressure 0
   // (This is appropriate since this is currently an incompressible
   // model)
   const double c1 = -9.20601e-2; // constant term
   const double c2 =  5.10768e-2; // T term
   const double c3 =  8.05999e-1; // S term
   const double c4 = -7.40849e-3; // T^2 term
   const double c5 = -3.01036e-3; // ST term
   const double c6 =  3.32267e-5; // T^3 term
   const double c7 =  3.21931e-5; // ST^2 term

   return c1 + c2*T + c3*S + c4*T*T + c5*S*T + c6*T*T*T + c7*S*T*T;
}

// Define a Blitz-friendly operator
BZ_DECLARE_FUNCTION2(eqn_of_state)

inline double eqn_of_state_t(double T){
   // Specialize for freshwater with S=0
   return eqn_of_state(T,0.0);
}
BZ_DECLARE_FUNCTION(eqn_of_state_t)

#endif
