#include "TArray.hpp"
#include <blitz/array.h>
#include "T_util.hpp"
#include "Par_util.hpp"
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "Splits.hpp"
#include <random/normal.h>

using namespace TArrayn;
using namespace Transformer;

using blitz::Array;
using blitz::TinyVector;
using blitz::GeneralArrayStorage;

using ranlib::Normal;

using namespace std;

// Defines for physical parameters

#define G (9.81)
// NON-ROTATIONAL
#define EARTH_OMEGA (0*2*M_PI/(24*3600))
#define EARTH_RADIUS (6371e3)
#define LATITUDE (M_PI/2)
#define ROT_F (0*2*EARTH_OMEGA*sin(LATITUDE))
#define ROT_B (0*2*EARTH_OMEGA*cos(LATITUDE)/EARTH_RADIUS)

// Grid size

#define Nx 512
#define Ny 512

// Grid lengths

#define Lx 4e5
#define Ly 4e5

// Normalization factors

#define Norm_x (2*M_PI/Lx)
#define Norm_y (2*M_PI/Ly)

// Depths

#define H0 (100.0)

// Long-wave speed

#define C0 (sqrt(G*H0))

// Timestep parameters

#define FINAL_T 16000 //0.5*24*3600.0
#define INITIAL_T 0.0
#define SAFETY_FACTOR 0.25

// Blitz index placeholders

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

// Defines for matrix entries

#define A11 (1+kvec(ii)*kvec(ii)*H0*H0/6)
#define A12 (H0*H0/6*kvec(ii)*lvec(kk))
#define A21 (A12)
#define A22 (1+H0*H0/6*lvec(kk)*lvec(kk))
#define MYDET (A11*A22-A21*A12)
#define B11 (A22/MYDET)
#define B22 (A11/MYDET)
#define B12 (-A12/MYDET)
#define B21 (-A21/MYDET)

#define numsteps 50


int main(int argc, char ** argv) {

   // Initialize MPI
   MPI_Init(&argc, &argv);

   // Allocate space for u, v, and eta levels
   vector<DTArray *> u_levels(3);
   vector<DTArray *> v_levels(3);
   vector<DTArray *> eta_levels(3);
   DTArray *pforce_ar, *pforce_t_ar;

   TinyVector<int,3> local_lbound, local_extent;
   GeneralArrayStorage<3> local_storage;

   // Get parameters for local array storage

   local_lbound = alloc_lbound(Nx,1,Ny);
   local_extent = alloc_extent(Nx,1,Ny);
   local_storage = alloc_storage(Nx,1,Ny);

   // Allocate the arrays used in the above
   u_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   u_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   u_levels[2] = new DTArray(local_lbound,local_extent,local_storage);
   v_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   v_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   v_levels[2] = new DTArray(local_lbound,local_extent,local_storage);
   eta_levels[0] = new DTArray(local_lbound,local_extent,local_storage);
   eta_levels[1] = new DTArray(local_lbound,local_extent,local_storage);
   eta_levels[2] = new DTArray(local_lbound,local_extent,local_storage);
   pforce_ar = new DTArray(local_lbound,local_extent,local_storage);
   pforce_t_ar = new DTArray(local_lbound,local_extent,local_storage);

   // Necessary FFT Transformers

   Trans1D X_xform(Nx,1,Ny,firstDim,FOURIER),
           Y_xform(Nx,1,Ny,thirdDim,FOURIER);

   TransWrapper XY_xform(Nx,1,Ny,FOURIER,NONE,FOURIER);

   // Grid in x, y

   Array<double,1> xgrid(split_range(Nx)), ygrid(Ny);

   xgrid = (ii+0.5)/Nx*Lx - Lx/2;
   ygrid = (ii+0.5)/Ny*Ly - Ly/2;

   // K, L vectors

   Array<double,1> kvec(XY_xform.wavenums(firstDim)), 
                  lvec(XY_xform.wavenums(thirdDim));
   kvec = kvec*Norm_x;
   lvec = lvec*Norm_y;
   

   // Compute timestep
   //double dt = SAFETY_FACTOR*fmin(Lx/Nx,Ly/Ny)/C0;
   double dt = 1;
   double t = INITIAL_T;
   if (master()) printf("Using timestep of %gs, with final time of %gs\n",dt,FINAL_T*1.0);
   if (master()) printf("Reference: C_0 = %g m/s, dx = %g m, dy = %g m\n",C0,Lx/Nx,Ly/Ny);

   // apply initial conditions
   *u_levels[1] = 0;
   //*u_levels[1] = 2*cos(0.5*M_PI*ygrid(kk)/Ly)/cosh(ygrid(kk)/100);
   int myrank;
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   Normal<double> rnd(0,1); rnd.seed(myrank);
   /*for (int i = u_levels[1]->lbound(firstDim); i <= u_levels[1]->ubound(firstDim); i++) {
      rnd.seed(i);
      for (int k = u_levels[1]->lbound(thirdDim); k <= u_levels[1]->ubound(thirdDim); k++)
         (*u_levels[1])(i,0,k) *= (1+1e-6*rnd.random());
   }*/
   *v_levels[1] = 0;
   *v_levels[1] = -cos(2*M_PI*xgrid(ii)/Lx)/pow(cosh((xgrid(ii)/Lx*10)),2);
   double IC_SCALE = sqrt(2*C0/ROT_B);
//   *eta_levels[1] = 0.75*H0*exp(-pow(ygrid(kk)/IC_SCALE,2))/cosh(xgrid(ii)/(IC_SCALE));
   *eta_levels[1] = 0;
//   *eta_levels[1] = 0.5*H0*exp(-(xgrid(ii)*xgrid(ii)+ygrid(kk)*ygrid(kk))/(0.01*Lx*Lx));
   // Allocating some physical-space temporaries
   DTArray temp1(local_lbound,local_extent,local_storage),
           temp2(local_lbound,local_extent,local_storage);

   // Iteration count
   int iterct = 0;
   *u_levels[2] = xgrid(ii)+0*kk;
   write_array(*u_levels[2],"xgrid");
   write_reader(*u_levels[2],"xgrid");
   *u_levels[2] = ygrid(kk);
   write_array(*u_levels[2],"ygrid");
   write_reader(*u_levels[2],"ygrid");
   
   write_reader(*u_levels[1],"u",true);
   write_reader(*v_levels[1],"v",true);
   write_reader(*eta_levels[1],"eta",true);
   write_array(*u_levels[1],"u",0);
   write_array(*v_levels[1],"v",0);
   write_array(*eta_levels[1],"eta",0);

   write_reader(*eta_levels[1],"pforce",true);
   write_reader(*eta_levels[1],"pforcet",true);
   {  // Euler step
      // Make nice references for arrays used
      DTArray &u_n = *u_levels[1], &v_n = *v_levels[1], &eta_n = *eta_levels[1],
              &u_p = *u_levels[2], &v_p = *v_levels[2], &eta_p = *eta_levels[2],
              &temp = temp1, &rhs = temp2;


      // u:
      // u_p = u_n - dt*(u_n u_n_x + v_n u_n_y + G eta_n_x)
      // Calculate u_n_x
      deriv_fft(u_n,X_xform,temp);
      rhs = u_n * temp * Norm_x;
      // u_n_y
      deriv_fft(u_n,Y_xform,temp);
      rhs = rhs + v_n*temp * Norm_y;
      // G eta_n_x
      deriv_fft(eta_n,X_xform,temp);
      u_p = u_n - dt*(rhs + G*temp*Norm_x - (ROT_F + ROT_B*ygrid(kk))*v_n);

      // v:
      // v_p = v_n - dt*(u_n v_n_x + v_n v_n_y + G eta_n_y)
      // Calculate v_n_x
      deriv_fft(v_n,X_xform,temp);
      rhs = u_n * temp * Norm_x;
      // v_n_y
      deriv_fft(v_n,Y_xform,temp);
      rhs = rhs + v_n*temp * Norm_y;
      // G eta_n_x
      deriv_fft(eta_n,Y_xform,temp);
      v_p = v_n - dt*(rhs + G*temp * Norm_y + (ROT_F+ROT_B*ygrid(kk))*u_n);

      // eta:
      // eta_p = eta_n - dt*((eta_n u)_x + (eta_n v)*y)
      temp = (H0+eta_n)*u_n;
      deriv_fft(temp,X_xform,temp);
      rhs = temp * Norm_x;
      temp = (H0+eta_n)*v_n;
      deriv_fft(temp,Y_xform,temp);
      eta_p = eta_n - dt*(temp*Norm_y+rhs);
      

      // Update t and cycle arrays
      t = t + dt;
      DTArray * tmp ;
      tmp = u_levels[0];
      u_levels[0] = u_levels[1];
      u_levels[1] = u_levels[2];
      u_levels[2] = tmp;
      tmp = v_levels[0];
      v_levels[0] = v_levels[1];
      v_levels[1] = v_levels[2];
      v_levels[2] = tmp;
      tmp = eta_levels[0];
      eta_levels[0] = eta_levels[1];
      eta_levels[1] = eta_levels[2];
      eta_levels[2] = tmp;
      

      iterct++;
      if (master()) printf("Completed time %g (iter %d)\n",t,iterct);
      double mu = psmax(max(abs(u_p))), mv = psmax(max(abs(v_p))), meta = pvmax(eta_p);
      if (master()) printf("Max u %g, v %g, eta %g\n",mu,mv,meta);
   }

   // Leapfrog steps
   // The leapfrog steps are done in spectral space, because
   // the velocities couple on a per-wavenumber basis.
   
   // Spectral temporaries for the RHS
   GeneralArrayStorage<3> spec_ordering;
   spec_ordering.ordering()[0] = XY_xform.get_complex_temp()->ordering(0);
   spec_ordering.ordering()[1] = XY_xform.get_complex_temp()->ordering(1);
   spec_ordering.ordering()[2] = XY_xform.get_complex_temp()->ordering(2);
   TinyVector<int,3> spec_lbound, spec_extent;
   spec_lbound = XY_xform.get_complex_temp()->lbound();
   spec_extent = XY_xform.get_complex_temp()->extent();

   CTArray rhs1(spec_lbound,spec_extent,spec_ordering),
           rhs2(spec_lbound,spec_extent,spec_ordering);

   // Normalization factor for the 2D transform
   double norm_2d = XY_xform.norm_factor();

   while (t < FINAL_T) {
      // Make nice references for arrays used
      DTArray &u_m = *u_levels[0], &v_m = *v_levels[0], &eta_m = *eta_levels[0],
              &u_n = *u_levels[1], &v_n = *v_levels[1], &eta_n = *eta_levels[1],
              &u_p = *u_levels[2], &v_p = *v_levels[2], &eta_p = *eta_levels[2],
              &pforce = *pforce_ar, &pforce_t = *pforce_t_ar;

      // RHS1 = FFT_XY(u ux + v uy + g etax)
      deriv_fft(u_n,X_xform,temp1);
      temp2 = u_n*temp1*Norm_x;
      deriv_fft(u_n,Y_xform,temp1);
      temp2 = temp2 + v_n*temp1*Norm_y;
      deriv_fft(eta_n,X_xform,temp1);
      temp2 = temp2 + G*temp1*Norm_x;
      // Spectral transform
      XY_xform.forward_transform(&temp2,FOURIER,NONE,FOURIER);
      rhs1 = *(XY_xform.get_complex_temp())/norm_2d;

      // Repeat for v
      // RHS2 = FFT_XY(u vx + v vy + g etax)
      deriv_fft(v_n,X_xform,temp1);
      temp2 = u_n*temp1*Norm_x;
      deriv_fft(v_n,Y_xform,temp1);
      temp2 = temp2 + v_n*temp1*Norm_y;
      deriv_fft(eta_n,Y_xform,temp1);
      temp2 = temp2 + G*temp1*Norm_y;
      // Spectral transform
      XY_xform.forward_transform(&temp2,FOURIER,NONE,FOURIER);
      rhs2 = *(XY_xform.get_complex_temp())/norm_2d;

      // Now, with rhs1/2, solve for u_rhs
      // (u_p = u_m - 2*dt*u_rhs) in physical space
      *(XY_xform.get_complex_temp()) = -2*dt*(B11*rhs1 + B12*rhs2);
      XY_xform.back_transform(&u_p,FOURIER,NONE,FOURIER);
      u_p = u_p + u_m + 2*dt*(ROT_F+ROT_B*ygrid(kk))*v_n;

      // Repeat for v_rhs
      *(XY_xform.get_complex_temp()) = -2*dt*(B21*rhs1 + B22*rhs2);
      XY_xform.back_transform(&v_p,FOURIER,NONE,FOURIER);
      v_p = v_p + v_m - 2*dt*(ROT_F+ROT_B*ygrid(kk))*u_n;

      // eta update
      // Pressure forcing
      const double p0 = 101; // Mean sea level pressure
      const double delta_p = 3; // Maximum pressure drop
      const double storm_r = 10e3; // Radius of storm
      double p_xs = 1.0*C0*t/sqrt(2)-Lx/2+2*storm_r; // x-centre of storm
      double p_ys = 1.0*C0*t/sqrt(2)-Ly/2+2*storm_r; // y-centre of storm
      pforce= - delta_p*exp(-(pow(xgrid(ii)-p_xs,2) + pow(ygrid(kk)-p_ys,2))/(storm_r*storm_r));
      pforce_t = -sqrt(2)*C0/(storm_r*storm_r*G) * (xgrid(ii)-p_xs + ygrid(kk)-p_ys) * pforce;
      temp1 = (H0+eta_n)*u_n;
      deriv_fft(temp1,X_xform,temp1);
      eta_p = -2*dt*temp1*Norm_x;
      temp1 = (H0+eta_n)*v_n;
      deriv_fft(temp1,Y_xform,temp1);
      eta_p = eta_p - 2*dt*temp1*Norm_y + eta_m + 2*dt*pforce_t;

      // Filter, with sensible defaults
      filter3(u_p,XY_xform,FOURIER,NONE,FOURIER,0.6,2,20);
      filter3(v_p,XY_xform,FOURIER,NONE,FOURIER,0.6,2,20);
      filter3(eta_p,XY_xform,FOURIER,NONE,FOURIER,0.6,2,20);
      

      iterct++;
      if (!(iterct % numsteps)) {
         write_array(u_p,"u",iterct/numsteps);
         write_array(v_p,"v",iterct/numsteps);
         write_array(eta_p,"eta",iterct/numsteps);
         write_array(pforce,"pforce",iterct/numsteps);
         write_array(pforce_t,"pforcet",iterct/numsteps);
      }
      // Update t and cycle arrays
      t = t + dt;
      DTArray * tmp ;
      tmp = u_levels[0];
      u_levels[0] = u_levels[1];
      u_levels[1] = u_levels[2];
      u_levels[2] = tmp;
      tmp = v_levels[0];
      v_levels[0] = v_levels[1];
      v_levels[1] = v_levels[2];
      v_levels[2] = tmp;
      tmp = eta_levels[0];
      eta_levels[0] = eta_levels[1];
      eta_levels[1] = eta_levels[2];
      eta_levels[2] = tmp;

         

      if (!(iterct % numsteps) || iterct < 20) {
         if (master()) printf("Completed time %g (iter %d)\n",t,iterct);
         double mu = psmax(max(abs(u_p))), mv = psmax(max(abs(v_p))), meta = pvmax(eta_p);
         if (master()) printf("Max u %g, v %g, eta %g\n",mu,mv,meta);
      }
   }   
      /*write_array(*u_levels[1],"u",iterct/numsteps+1);
      write_array(*v_levels[1],"v",iterct/numsteps+1);
      write_array(*eta_levels[1],"eta",iterct/numsteps+1);*/
    
   if (master()) printf("Finished at time %g!\n",t);  


   MPI_Finalize();
   return 0;
}


   
