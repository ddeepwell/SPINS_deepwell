/* WARNING: Science Content!

   Implementation of various analysis routines */

#include "Science.hpp"
#include "math.h"
#include "Par_util.hpp"
#include "stdio.h"
#include "Split_reader.hpp"
#include "T_util.hpp"
#include "Parformer.hpp"

// Marek's Overturning Diagnostic

using blitz::Array;
using blitz::cos;
using namespace TArrayn;
using namespace NSIntegrator;
using namespace Transformer;


/* Marek's overturning diagnotic.  For a given density, compute the total sum
   of z-levels (by vertical column) for which the density is statically 
   unstable.  That is, a completely stable stratification will return an Array
   full of 0s, and a completely unstable stratification will return an Array
   full of (zmax - zmin). */
Array<double,3> overturning_2d(
      Array<double,3> const & rho, // Density
      Array<double,1> const & zgrid, // Z-levels
      Dimension reduce // Dimension to consider the vertical
      ) {
   using namespace TArrayn;
   blitz::RectDomain<3> dims = rho.domain();
   // Now, for general behaviour over reduced dimensions, figure out min/max
   // for the output Array (and for the iteration inside)
   int szxmin, szymin, szzmin, szxmax, szymax, szzmax;
   szxmin = dims.lbound(firstDim);
   szymin = dims.lbound(secondDim);
   szzmin = dims.lbound(thirdDim);
   szxmax = dims.ubound(firstDim);
   szymax = dims.ubound(secondDim);
   szzmax = dims.ubound(thirdDim);

   // Assert that the currently-split dimension is fully available
   assert(dims.lbound(reduce) == 0);
   // Now, reset the "max" of the reduced dimension to the "min"
   switch(reduce) {
      case firstDim:
         szxmax = szxmin; break;
      case secondDim:
         szymax = szymin; break;
      case thirdDim:
         szzmax = szzmin; break;
   }
   // Define the output Array
   Array<double,3> output(blitz::Range(szxmin,szxmax), 
                          blitz::Range(szymin,szymax),
                          blitz::Range(szzmin,szzmax));

   // Now, loop over the output points and sum up the overturning water column
   double zdiff = zgrid(zgrid.ubound()) - zgrid(zgrid.lbound());
   // Calculate a threshold value -- otherwise simple rounding error can
   // cause erroneous overturning

   /* As an ad-hoc measure, set the threshold of "significant" overturning to
      the maximum of:
         1) 1e-8 * the maximum rho value, or
         2) an overturning of 1%, extended over the entire domain, that is
            2% * (max-min) * LZ / NZ */
     double maxrho = pvmax(rho);
    double minrho = pvmin(rho); 
   double thresh = fmax(1e-8*maxrho,fabs(zdiff) * (maxrho-minrho) * 1e-2
      / (zgrid.ubound(firstDim) - zgrid.lbound(firstDim)));
   for (int i = szxmin; i <= szxmax; i++) {
      for (int j = szymin; j <= szymax; j++) {
         for (int k = szzmin; k <= szzmax; k++) {
            /* Now, build a zplus/zminus pair of ranges for the density
               and z-level differencing, and do the reduction.  Most of
               the code duplication here arises because Blitz doesn't like
               sum() reduction over anything but the last logical dimension */
            if (reduce == firstDim) {
               blitz::Range zplus(rho.lbound(firstDim)+1,rho.ubound(firstDim));
               blitz::Range zminus(rho.lbound(firstDim),rho.ubound(firstDim)-1);
               output(i,j,k) = fabs(sum(
                     where(zdiff * (rho(zplus,j,k) - rho(zminus,j,k)) > thresh,
                        zgrid(zplus) - zgrid(zminus), 0)));
            } else if (reduce == secondDim) {
               blitz::Range zplus(rho.lbound(secondDim)+1,rho.ubound(secondDim));
               blitz::Range zminus(rho.lbound(secondDim),rho.ubound(secondDim)-1);
               output(i,j,k) = fabs(sum(
                     where(zdiff * (rho(i,zplus,k) - rho(i,zminus,k)) > thresh,
                        zgrid(zplus) - zgrid(zminus), 0)));
            } else if (reduce == thirdDim) {
               blitz::Range zplus(rho.lbound(thirdDim)+1,rho.ubound(thirdDim));
               blitz::Range zminus(rho.lbound(thirdDim),rho.ubound(thirdDim)-1);
               output(i,j,k) = fabs(sum(
                     where(zdiff * (rho(i,j,zplus) - rho(i,j,zminus)) > thresh,
                        zgrid(zplus) - zgrid(zminus), 0)));
            }
         }
      }
   }

   return output;
}


// Read in a 2D file and interpret it as a 2D slice of a 3D array, for
// initialization with read-in-data from a program like MATLAB
void read_2d_slice(Array<double,3> & fillme, const char * filename, 
                  int Nx, int Ny) {

   using blitz::ColumnMajorArray;
   using blitz::firstDim; using blitz::secondDim; using blitz::thirdDim;
   /* Get the local ranges we're interested in */
   blitz::Range xrange(fillme.lbound(firstDim),fillme.ubound(firstDim));
   blitz::Range zrange(fillme.lbound(thirdDim),fillme.ubound(thirdDim));
   
   /* Read the 2D slice from disk.  Matlab uses Column-Major array storage */

   blitz::GeneralArrayStorage<2> storage_order = blitz::ColumnMajorArray<2>();
   blitz::Array<double,2> * sliced = 
      read_2d_slice<double>(filename,Nx,Ny,xrange,zrange,storage_order);

   /* Now, assign the slice to fill the 3D array */
   for(int y = fillme.lbound(secondDim); y <= fillme.ubound(secondDim); y++) {
      fillme(xrange,y,zrange) = (*sliced)(xrange,zrange);
   }
   delete sliced; 
}
   

void vorticity(TArrayn::DTArray & u, TArrayn::DTArray & v, 
      TArrayn::DTArray & w, 
      TArrayn::DTArray * & vor_x, TArrayn::DTArray * & vor_y,
      TArrayn::DTArray * & vor_z, double Lx, double Ly, double Lz,
      int szx, int szy, int szz,
      NSIntegrator::DIMTYPE DIM_X, NSIntegrator::DIMTYPE DIM_Y, 
      NSIntegrator::DIMTYPE DIM_Z) {
   static int Nx = 0, Ny = 0, Nz = 0;
   static Transformer::Trans1D trans_x(szx,szy,szz,firstDim,
                     (DIM_X == PERIODIC ? FOURIER : REAL)),
                  trans_y(szx,szy,szz,secondDim,
                     (DIM_Y == PERIODIC ? FOURIER : REAL)),
                  trans_z (szx,szy,szz,thirdDim,
                     (DIM_Z == PERIODIC ? FOURIER : REAL));
   static blitz::TinyVector<int,3> 
      local_lbounds(alloc_lbound(szx,szy,szz)),
      local_extent(alloc_extent(szx,szy,szz));
   static blitz::GeneralArrayStorage<3> 
      local_storage(alloc_storage(szx,szy,szz));
   static DTArray vort_x(local_lbounds,local_extent,local_storage),
                  vort_y(local_lbounds,local_extent,local_storage),
                  vort_z(local_lbounds,local_extent,local_storage),
                  temp_a(local_lbounds,local_extent,local_storage);
   /* Initialization */
   if (Nx == 0 || Ny == 0 || Nz == 0) {
      Nx = szx; Ny = szy; Nz = szz;
   }
   assert (Nx == szx && Ny == szy && Nz == szz);
   /* x-vorticity is w_y - v_z */
   vort_x = 0;
   if (szy > 1) { // w_y
      if (DIM_X == PERIODIC) {
         deriv_fft(w,trans_y,temp_a);
         vort_x = temp_a*(2*M_PI/Ly);
      } else if (DIM_X == FREE_SLIP) {
         deriv_dct(w,trans_y,temp_a);
         vort_x = temp_a*(M_PI/Ly);
      } else {
         assert(DIM_X == NO_SLIP);
         deriv_cheb(w,trans_y,temp_a);
         vort_x = temp_a*(-2/Ly);
      }
   }
   if (szz > 1) { // v_z
      if (DIM_Z == PERIODIC) {
         deriv_fft(v,trans_z,temp_a);
         vort_x -= temp_a*(2*M_PI/Lz);
      } else if (DIM_Z == FREE_SLIP) {
         deriv_dct(v,trans_z,temp_a);
         vort_x -= temp_a*(M_PI/Lz);
      } else {
         assert(DIM_Z == NO_SLIP);
         deriv_cheb(v,trans_z,temp_a);
         vort_x -= temp_a*(-2/Lz);
      }
   }
   // y-vorticity is u_z - w_x
   vort_y = 0;
   if (szz > 1) { // u_z
      if (DIM_Z == PERIODIC) {
         deriv_fft(u,trans_z,temp_a);
         vort_y = temp_a*(2*M_PI/Lz);
      } else if (DIM_Z == FREE_SLIP) {
         deriv_dct(u,trans_z,temp_a);
         vort_y = temp_a*(M_PI/Lz);
      } else {
         assert(DIM_Z == NO_SLIP);
         deriv_cheb(u,trans_z,temp_a);
         vort_y = temp_a*(-2/Lz);
      }
   }
   if (szx > 1) { // w_x
      if (DIM_X == PERIODIC) {
         deriv_fft(w,trans_x,temp_a);
         vort_y -= temp_a*(2*M_PI/Lx);
      } else if (DIM_X == FREE_SLIP) {
         deriv_dct(w,trans_x,temp_a);
         vort_y -= temp_a*(M_PI/Lx);
      } else {
         assert(DIM_X == NO_SLIP);
         deriv_cheb(w,trans_x,temp_a);
         vort_y -= temp_a*(-2/Lx);
      }
   }
   // And finally, vort_z is v_x - u_y
   vort_z = 0;
   if (szx > 1) { // v_x
      if (DIM_X == PERIODIC) {
         deriv_fft(v,trans_x,temp_a);
         vort_z = temp_a*(2*M_PI/Lx);
      } else if (DIM_X == FREE_SLIP) {
         deriv_dct(v,trans_x,temp_a);
         vort_z = temp_a*(M_PI/Lx);
      } else {
         assert(DIM_X == NO_SLIP);
         deriv_cheb(v,trans_x,temp_a);
         vort_z = temp_a*(-2/Lx);
      }
   }
   if (szy > 1) { // u_y
      if (DIM_Y == PERIODIC) {
         deriv_fft(u,trans_y,temp_a);
         vort_z -= temp_a*(2*M_PI/Ly);
      } else if (DIM_Y == FREE_SLIP) {
         deriv_dct(u,trans_y,temp_a);
         vort_z -= temp_a*(M_PI/Ly);
      } else {
         assert(DIM_Y == NO_SLIP);
         deriv_cheb(u,trans_y,temp_a);
         vort_z -= temp_a*(-2/Ly);
      }
   }
   vor_x = &vort_x;
   vor_y = &vort_y;
   vor_z = &vort_z;
   return;
}

// Global arrays to store quadrature weights
Array<double,1> _quadw_x, _quadw_y, _quadw_z;

// Compute quadrature weights
void compute_quadweights(int szx, int szy, int szz, 
      double Lx, double Ly, double Lz,
      NSIntegrator::DIMTYPE DIM_X, NSIntegrator::DIMTYPE DIM_Y,
      NSIntegrator::DIMTYPE DIM_Z) {
   _quadw_x.resize(split_range(szx));
   _quadw_y.resize(szy); _quadw_z.resize(szz);
   if (DIM_X == NO_SLIP) {
      blitz::firstIndex ii;
      _quadw_x = 1;
      for (int k = 1; k <= (szx-2)/2; k++) {
         // From Trefethen, Spectral Methods in MATLAB
         // clenshaw-curtis quadrature weights
         _quadw_x -= 2*cos(2*k*M_PI*ii/(szx-1))/(4*k*k-1);
      }
      if ((szx%2))
         _quadw_x -= cos(M_PI*ii)/((szx-1)*(szx-1)-1);
      _quadw_x = 2*_quadw_x/(szx-1);
      if (_quadw_x.lbound(firstDim) == 0) {
         _quadw_x(0) = 1.0/(szx-1)/(szx-1);
      }
      if (_quadw_x.ubound(firstDim) == (szx-1)) {
         _quadw_x(szx-1) = 1.0/(szx-1)/(szx-1);
      }
      _quadw_x *= Lx/2;
   } else {
      // Trapezoid rule
      _quadw_x = Lx/szx;
   }
   if (DIM_Y == NO_SLIP) {
      blitz::firstIndex ii;
      _quadw_y = 1;
      for (int k = 1; k <= (szy-2)/2; k++) {
         // From Trefethen, Spectral Methods in MATLAB
         // clenshaw-curtis quadrature weights
         _quadw_y -= 2*cos(2*k*(M_PI*(ii)/(szy-1)))/(4*k*k-1);
      }
      if ((szy%2))
         _quadw_y -= cos(M_PI*ii)/((szy-1)*(szy-1)-1);
      _quadw_y = 2*_quadw_y/(szy-1);
      _quadw_y(0) = 1.0/(szy-1)/(szy-1);
      _quadw_y(szy-1) = 1.0/(szy-1)/(szy-1);
      _quadw_y *= Ly/2;
   } else {
      // Trapezoid rule
      _quadw_y = Ly/szy;
   }
   if (DIM_Z == NO_SLIP) {
      blitz::firstIndex ii;
      _quadw_z = 1;
      for (int k = 1; k <= (szz-2)/2; k++) {
         // From Trefethen, Spectral Methods in MATLAB
         // clenshaw-curtis quadrature weights
         _quadw_z -= 2*cos(2*k*(M_PI*(ii)/(szz-1)))/(4*k*k-1);
      }
      if ((szz%2))
         _quadw_z -= cos(M_PI*ii)/((szz-1)*(szz-1)-1);
      _quadw_z = 2*_quadw_z/(szz-1);
      _quadw_z(0) = 1.0/(szz-1)/(szz-1);
      _quadw_z(szz-1) = 1.0/(szz-1)/(szz-1);
      _quadw_z *= Lz/2;
   } else {
      // Trapezoid rule
      _quadw_z = Lz/szz;
   }
}
const blitz::Array<double,1> * get_quad_x() { return &_quadw_x;}
const blitz::Array<double,1> * get_quad_y() { return &_quadw_y;}
const blitz::Array<double,1> * get_quad_z() { return &_quadw_z;}
   

   
 
   
