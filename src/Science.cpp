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


// Read in a 2D-array from file and extend it to fill a full, 3D array in
// memory.  Unlike the following function, this uses the standard C storage
// order -- matlab uses the transpose of column-major ordering
void read_2d_restart(Array<double,3> & fillme, const char * filename, 
                  int Nx, int Ny) {

//   using blitz::ColumnMajorArray;
   using blitz::firstDim; using blitz::secondDim; using blitz::thirdDim;
   /* Get the local ranges we're interested in */
   blitz::Range xrange(fillme.lbound(firstDim),fillme.ubound(firstDim));
   blitz::Range zrange(fillme.lbound(thirdDim),fillme.ubound(thirdDim));
   
   /* Read the 2D slice from disk.  Matlab uses Column-Major array storage */

   blitz::GeneralArrayStorage<2> storage_order;
   blitz::Array<double,2> * sliced = 
      read_2d_slice<double>(filename,Nx,Ny,xrange,zrange,storage_order);

   /* Now, assign the slice to fill the 3D array */
   for(int y = fillme.lbound(secondDim); y <= fillme.ubound(secondDim); y++) {
      fillme(xrange,y,zrange) = (*sliced)(xrange,zrange);
   }
   delete sliced; 
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
   
// X-component of vorticity
void compute_vort_x(TArrayn::DTArray & v, TArrayn::DTArray & w, TArrayn::DTArray & vortx,
        TArrayn::Grad * gradient_op, const string * grid_type) {
    // Set-up
    S_EXP expan[3];
    assert(gradient_op);

    // Setup for dv/dz
    find_expansion(grid_type, expan, "v", "");
    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
    // get dv/dz
    gradient_op->get_dz(&vortx,false);
    // Invert to get the negative
    vortx = (-1)*vortx;

    // Setup for dw/dy
    find_expansion(grid_type, expan, "w", "");
    gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
    // get dw/dy, and add to vortx
    gradient_op->get_dy(&vortx,true);
}

// Y-component of vorticity
void compute_vort_y(TArrayn::DTArray & u, TArrayn::DTArray & w, TArrayn::DTArray & vorty,
       TArrayn::Grad * gradient_op, const string * grid_type) {
    // Set-up
    S_EXP expan[3];
    assert(gradient_op);

    // Setup for dw/dx
    find_expansion(grid_type, expan, "w", "");
    gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
    // get dw/dx
    gradient_op->get_dx(&vorty,false);
    // Invert to get the negative
    vorty = (-1)*vorty;

    // Setup for du/dz
    find_expansion(grid_type, expan, "u", "");
    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
    // get du/dz
    gradient_op->get_dz(&vorty,true);
}

// Z-component of vorticity
void compute_vort_z(TArrayn::DTArray & u, TArrayn::DTArray & v, TArrayn::DTArray & vortz,
       TArrayn::Grad * gradient_op, const string * grid_type) {
    // Set-up
    S_EXP expan[3];
    assert(gradient_op);

    // Setup for du/dy
    find_expansion(grid_type, expan, "u", "");
    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
    // get du/dy
    gradient_op->get_dy(&vortz,false);
    // Invert to get the negative
    vortz = (-1)*vortz;

    // Setup for dv/dx
    find_expansion(grid_type, expan, "v", "");
    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
    // get du/dz
    gradient_op->get_dx(&vortz,true);
}

// Vorticity
void compute_vorticity(TArrayn::DTArray & u, TArrayn::DTArray & v, TArrayn::DTArray & w,
        TArrayn::DTArray & vortx, TArrayn::DTArray & vorty, TArrayn::DTArray & vortz,
        TArrayn::Grad * gradient_op, const string * grid_type) {
    // compute each component
    compute_vort_x(v, w, vortx, gradient_op, grid_type);
    compute_vort_y(u, w, vorty, gradient_op, grid_type);
    compute_vort_z(u, v, vortz, gradient_op, grid_type);
}

// Enstrophy Density: 1/2*(vort_x^2 + vort_y^2 + vort_z^2)
void enstrophy_density(TArrayn::DTArray & u, TArrayn::DTArray & v, TArrayn::DTArray & w,
        TArrayn::DTArray & enst, TArrayn::Grad * gradient_op, const string * grid_type,
        const int Nx, const int Ny, const int Nz) {
    // initalize temporary array
    static DTArray *temp = alloc_array(Nx,Ny,Nz);

    // square vorticity components
    compute_vort_x(v, w, *temp, gradient_op, grid_type);
    enst = pow(*temp,2);
    compute_vort_y(u, w, *temp, gradient_op, grid_type);
    enst += pow(*temp,2);
    compute_vort_z(u, v, *temp, gradient_op, grid_type);
    enst += pow(*temp,2);
    enst = 0.5*enst;
}

// Viscous dissipation: 2*mu*e_ij*e_ij
void dissipation(TArrayn::DTArray & u, TArrayn::DTArray & v, TArrayn::DTArray & w,
        TArrayn::DTArray & diss, TArrayn::Grad * gradient_op, const string * grid_type,
        const int Nx, const int Ny, const int Nz, const double visco) {
    // Set-up
    static DTArray *temp = alloc_array(Nx,Ny,Nz);
    S_EXP expan[3];
    assert(gradient_op);

    // 1st term: e_11^2 = (du/dx)^2
    find_expansion(grid_type, expan, "u", "");
    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
    gradient_op->get_dx(temp,false);
    diss = pow(*temp,2);
    // 2nd term: e_22^2 = (dv/dy)^2
    find_expansion(grid_type, expan, "v", "");
    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
    gradient_op->get_dy(temp,false);
    diss += pow(*temp,2);
    // 3rd term: e_33^2 = (dw/dz)^2
    find_expansion(grid_type, expan, "w", "");
    gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
    gradient_op->get_dz(temp,false);
    diss += pow(*temp,2);
    // 4th term: 2e_12^2 = 2*(1/2*(u_y + v_x))^2
    // u_y
    find_expansion(grid_type, expan, "u", "");
    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
    gradient_op->get_dy(temp,false);
    // v_x
    find_expansion(grid_type, expan, "v", "");
    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
    gradient_op->get_dx(temp,true);
    diss += 2.0*pow(0.5*(*temp),2);
    // 5th term: 2e_13^2 = 2*(1/2*(u_z + w_x))^2
    // u_z
    find_expansion(grid_type, expan, "u", "");
    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
    gradient_op->get_dz(temp,false);
    // w_x
    find_expansion(grid_type, expan, "w", "");
    gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
    gradient_op->get_dx(temp,true);
    diss += 2.0*pow(0.5*(*temp),2);
    // 6th term: 2e_23^2 = 2*(1/2*(v_z + w_y))^2
    // v_z
    find_expansion(grid_type, expan, "v", "");
    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
    gradient_op->get_dz(temp,false);
    // w_y
    find_expansion(grid_type, expan, "w", "");
    gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
    gradient_op->get_dy(temp,true);
    diss += 2.0*pow(0.5*(*temp),2);
    // multiply by 2*mu
    diss *= 2.0*visco;
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
const blitz::Array<double,1> * get_quad_x() { 
   // Check whether the quad weight has been initialized
   if (_quadw_x.length(firstDim) <= 0) {
      assert(0 && "Error: quadrature weights were not initalized before use");
   }
   return &_quadw_x;
}
const blitz::Array<double,1> * get_quad_y() {
   if (_quadw_y.length(firstDim) <= 0) {
      assert(0 && "Error: quadrature weights were not initalized before use");
   }
   return &_quadw_y;
}
const blitz::Array<double,1> * get_quad_z() {
   if (_quadw_z.length(firstDim) <= 0) {
      assert(0 && "Error: quadrature weights were not initalized before use");
   }
   return &_quadw_z;
}

// function to parse the expansion types
void find_expansion(const string * grid_type, S_EXP * expan,
        string deriv_filename, string base_field) {
    const int x_ind = 0;
    const int y_ind = 1;
    const int z_ind = 2;

    for ( int nn = 0; nn <= 2; nn++ ) {
        if      (grid_type[nn] == "FOURIER") { expan[nn] = FOURIER; }
        else if (grid_type[nn] == "NO_SLIP") { expan[nn] = CHEBY; }
        else if (grid_type[nn] == "FREE_SLIP") { 
            // setup for a first derivative 
            if ( deriv_filename == "u" or base_field == "u" ) {
                if      ( nn == x_ind ) { expan[nn] = SINE; }
                else if ( nn == y_ind ) { expan[nn] = COSINE; }
                else if ( nn == z_ind ) { expan[nn] = COSINE; }
            }
            else if ( deriv_filename == "v" or base_field == "v" ) {
                if      ( nn == x_ind ) { expan[nn] = COSINE; }
                else if ( nn == y_ind ) { expan[nn] = SINE; }
                else if ( nn == z_ind ) { expan[nn] = COSINE; }
            }
            else if ( deriv_filename == "w" or base_field == "w") {
                if      ( nn == x_ind ) { expan[nn] = COSINE; }
                else if ( nn == y_ind ) { expan[nn] = COSINE; }
                else if ( nn == z_ind ) { expan[nn] = SINE; }
            }
            else {
                if      ( nn == x_ind ) { expan[nn] = COSINE; }
                else if ( nn == y_ind ) { expan[nn] = COSINE; }
                else if ( nn == z_ind ) { expan[nn] = COSINE; }
            }
        }   
    }
}
// function to switch trig functions
S_EXP swap_trig( S_EXP the_exp ) {
    if ( the_exp == SINE ) {
        return COSINE; }
    else if ( the_exp == COSINE ) {
        return SINE; }
    else if ( the_exp == FOURIER ) {
        return FOURIER; }
    else if ( the_exp == CHEBY ) {
        return CHEBY; }
    else {
        MPI_Finalize(); exit(1); // stop
    }
}


// Bottom slope
void bottom_slope(TArrayn::DTArray & Hprime, TArrayn::DTArray & zgrid,
        TArrayn::DTArray & temp, TArrayn::Grad * gradient_op,
        const string * grid_type, const int Nx, const int Ny, const int Nz) {
    // Set-up
    DTArray & z_x = *alloc_array(Nx,Ny,Nz);
    blitz::Range all = blitz::Range::all();
    blitz::firstIndex ii;
    blitz::secondIndex jj;
    blitz::thirdIndex kk;
    S_EXP expan[3];
    assert(gradient_op);
    
    // get bottom topography
    Array<double,1> xx(split_range(Nx));
    xx = zgrid(all,0,Nz-1);
    // put into temp array, and take derivative
    temp = xx(ii) + 0*jj + 0*kk;
    find_expansion(grid_type, expan, "zgrid", "");
    gradient_op->setup_array(&temp,expan[0],expan[1],expan[2]);
    gradient_op->get_dx(&z_x);
    // flatten to get 2D array
    Hprime(all,all,0) = z_x(all,all,0);
    delete &z_x, xx;
}

// Top Stress (along "topography" - x)
void top_stress_x(TArrayn::DTArray & stress_x, TArrayn::DTArray & u,
        TArrayn::DTArray & temp, TArrayn::Grad * gradient_op,
        const string * grid_type, const double visco) {
    // Set-up
    blitz::Range all = blitz::Range::all();
    S_EXP expan[3];
    assert(gradient_op);

    // du/dz
    find_expansion(grid_type, expan, "u", "");
    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
    gradient_op->get_dz(&temp,false);
    // top stress
    stress_x(all,all,0) = visco*temp(all,all,0);
}
// Top Stress (across "topography" - y)
void top_stress_y(TArrayn::DTArray & stress_y, TArrayn::DTArray & v,
        TArrayn::DTArray & temp, TArrayn::Grad * gradient_op,
        const string * grid_type, const double visco) {
    // Set-up
    blitz::Range all = blitz::Range::all();
    S_EXP expan[3];
    assert(gradient_op);

    // dv/dz
    find_expansion(grid_type, expan, "v", "");
    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
    gradient_op->get_dz(&temp,false);
    // top stress
    stress_y(all,all,0) = visco*temp(all,all,0);
}
// Bottom Stress (along topography - x)
void bottom_stress_x(TArrayn::DTArray & stress_x, TArrayn::DTArray & Hprime,
        TArrayn::DTArray & u, TArrayn::DTArray & w, TArrayn::DTArray & temp,
        TArrayn::Grad * gradient_op, const string * grid_type, const bool mapped,
        const int Nz, const double visco) {
    // Set-up
    blitz::Range all = blitz::Range::all();
    S_EXP expan[3];
    assert(gradient_op);

    if (mapped) {
        // du/dx
        find_expansion(grid_type, expan, "u", "");
        gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
        gradient_op->get_dx(&temp,false);
        temp = (-1)*temp;
        // dw/dz
        find_expansion(grid_type, expan, "w", "");
        gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
        gradient_op->get_dz(&temp,true);
        // 2H'*(w_z-u_x)
        stress_x(all,all,0) = 2*Hprime(all,all,0)*temp(all,all,Nz-1);

        // dw/dx
        find_expansion(grid_type, expan, "w", "");
        gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
        gradient_op->get_dx(&temp,false);
        // du/dz
        find_expansion(grid_type, expan, "u", "");
        gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
        gradient_op->get_dz(&temp,true);
        // (1-(H')^2)*(u_z+w_x)
        stress_x(all,all,0) += (1-pow(Hprime(all,all,0),2))*temp(all,all,Nz-1);
        // multiply by mu/(1+(H')^2)
        stress_x = visco/(1+pow(Hprime,2))*stress_x;
    } else {
        // du/dz
        find_expansion(grid_type, expan, "u", "");
        gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
        gradient_op->get_dz(&temp,false);
        // top stress
        stress_x(all,all,0) = visco*temp(all,all,Nz-1);
    }
}
// Bottom Stress (across topography - y)
void bottom_stress_y(TArrayn::DTArray & stress_y, TArrayn::DTArray & Hprime,
        TArrayn::DTArray & v, TArrayn::DTArray & temp,
        TArrayn::Grad * gradient_op, const string * grid_type, const bool mapped,
        const int Nz, const double visco) {
    // Set-up
    blitz::Range all = blitz::Range::all();
    S_EXP expan[3];
    assert(gradient_op);

    if (mapped) {
        // dv/dx
        find_expansion(grid_type, expan, "v", "");
        gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
        gradient_op->get_dx(&temp,false);
        // -v_x*H'
        stress_y(all,all,0) = -temp(all,all,Nz-1)*Hprime(all,all,0);
        // dv/dz
        gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
        gradient_op->get_dz(&temp,false);
        // add to -v_x*H'
        stress_y(all,all,0) = temp(all,all,Nz-1) + stress_y(all,all,0);
        // multiply by mu/(1+(H')^2)
        stress_y = visco/(1+pow(Hprime,2))*stress_y;
    } else {
        // dv/dz
        find_expansion(grid_type, expan, "v", "");
        gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
        gradient_op->get_dz(&temp,false);
        // top stress
        stress_y(all,all,0) = visco*temp(all,all,Nz-1);
    }
}
