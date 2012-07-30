/* Set up for Nancy Soontiens, this case this is flow over a hill with periodic BCs and 3d*/

// Required headers 
#include <blitz/array.h> // Blitz++ array library
#include "../TArray.hpp" // Custom extensions to the library to support FFTs
#include "../NSIntegrator.hpp" // Time-integrator for the Navier-Stokes equations
#include <mpi.h> // MPI parallel library
#include "../BaseCase.hpp" // Support file that contains default implementations of several functions
#include "../T_util.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <random/normal.h>
#include <vector>


using namespace std;
using namespace NSIntegrator;
using namespace ranlib;
using namespace NSIntegrator;


// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

// Physical constants
const double g = 9.81;
const double rho_0 = 1; //1028; // Units of kg / m^3
const double nu = 1e-5; 

// Pysical parameters
const double ROT_F = 0; //0.5e-4; // 1/s, mid-latitude
const double LENGTH_X = 5; // 5m long
const double LENGTH_Z =  1; // 1m tall
const double LENGTH_Y = 1;  //1m wide

const double LINEAR_STRAT = 0.05/LENGTH_Z; // Linear component of stratification

//Forcing Parameters
const double F_U0=0.1;
const double F_T0=10;

// Define a helper function to give stratification
double inline background_strat(double z) {
   return rho_0 - LINEAR_STRAT*z;
}
// And declare it to the array library, so we can use really nifty
// array notation with it
BZ_DECLARE_FUNCTION(background_strat)

// Crater parameters
const double C_DEPTH = 0.2*LENGTH_Z; // Crater depth0.2m
const double C_LENGTH = LENGTH_X/4; // Crater width


// Numerical parameters
int NZ =150; // Number of vertical points.  Number of horizontal points
int NX =1000; // will be calculated based on this.
int NY = 10;

const double plot_interval = 1; 
const double final_time = 40; 

const double dt_max = 1; 


class mapiw : public BaseCase {
   public:
      // Variables to set the plot sequence number and time of the last writeout
      DTArray * xgrid,* ygrid, * zgrid;
      Array<double,1> hill;
      int plot_number; double last_plot;

      // Resolution in X, Y, and Z
      int size_x() const { return NX; }
      int size_y() const { return NY; }
      int size_z() const { return NZ; }

      /* Periodic in x and NoSLIP z, periodic in y if we have it */
      DIMTYPE type_z() const {return NO_SLIP;}
      DIMTYPE type_x() const {return PERIODIC;}
      DIMTYPE type_y() const {return PERIODIC;}
      DIMTYPE type_default() const { return PERIODIC; }

      /* The grid size is governed through the #defines above */
      double length_x() const { return LENGTH_X; }
      double length_y() const { return LENGTH_Y; }
      double length_z() const { return LENGTH_Z; }

      bool is_mapped() const {return true;}
      void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
         xgrid = alloc_array(NX,NY,NZ);
         zgrid = alloc_array(NX,NY,NZ);
         ygrid = alloc_array(NX,NY,NZ);
         Array<double,1> xx(split_range(NX)),yy(NY), zz(NZ);
         // Use periodic coordinates in horizontal
         xx = -LENGTH_X/2 + LENGTH_X*((0.5+ii)/NX); // Periodic grid in horizontal
         zz = cos(ii*M_PI/(NZ-1)); // Chebyshev in vertical
         yy = -LENGTH_Y/2 +LENGTH_Y*((0.5+ii)/NY);
         xg = xx(ii) + 0*jj + 0*kk;
         *xgrid = xg;

         //hill = H_HEIGHT/(1+pow(xx(ii)/H_LENGTH,2));
	 hill = C_DEPTH*exp(-pow(xx(ii)/(C_LENGTH/4),2));

         /* Now, for the z-grid, we want the top level to always
            be at the maximum height of LENGTH_Z.  The bottom level
            should always be at hill(x).  That means the vertical
            extent is LENGTH_Z-hill(x) */

         zg = hill(ii) + (LENGTH_Z-hill(ii))/2*(1+zz(kk));
         *zgrid = zg;

         yg = yy(jj) + 0*ii +0*kk;
         *ygrid = yg;

         write_array(xg,"xgrid");
         write_reader(xg,"xgrid",false);
         write_array(yg,"ygrid");
         write_reader(yg,"ygrid",false);
         write_array(zg,"zgrid");
         write_reader(zg,"zgrid",false);
      }
      /* Use one actively-modified tracer */
      int numActive() const { return 1; }

      // Use 10^-5 viscosity and diffusivity
      double get_visco() const { return nu; }
      double get_diffusivity(int t_num) const { return nu; }

      /* Start at t=0 */
      double init_time() const { return 0; }

      /* Modify the timestep if necessary in order to land evenly on a plot time */
      double check_timestep (double intime, double now) {
         // Firstly, the buoyancy frequency provides a timescale that is not
         // accounted for with the velocity-based CFL condition.
         if (intime > dt_max) {
            intime = dt_max;
         }
         // Now, calculate how many timesteps remain until the next writeout
         double until_plot = last_plot + plot_interval - now;
         int steps = ceil(until_plot / intime);
         // And calculate where we will actually be after (steps) timesteps
         // of the current size
         double true_fintime = steps*intime;

         // If that's close enough to the real writeout time, that's fine.
         if (fabs(until_plot - true_fintime) < 1e-6) {
            return intime;
         } else {
            // Otherwise, square up the timeteps.  This will always shrink the timestep.
            return (until_plot / steps);
         }
      }


      /* Initialize velocities at the start of the run.  For this simple
         case, initialize u=0 +randoms; */
      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
                  Array<double,1> xx(split_range(NX)), yy(NY), zz(NZ);
         xx = -(LENGTH_X/2) + LENGTH_X*(ii+0.5)/NX;
         yy = -(LENGTH_Y/2) + LENGTH_Y*(ii+0.5)/NY;
         zz = cos(ii*M_PI/(NZ-1));

         u = 0*ii +0*jj +0*kk;
         v = 0;
         w = 0;
         /* Add random initial perturbation */
         int myrank;
         MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
         /* Add random noise about 3.5 orders of magnitude below background*/
         Normal<double> rnd(0,1);
         rnd.seed(myrank);
         for (int i = u.lbound(firstDim); i<= u.ubound(firstDim); i++) {
            for (int j = u.lbound(secondDim); j<= u.ubound(secondDim); j++) {
               for (int k = u.lbound(thirdDim); k<= u.ubound(thirdDim); k++) {
                  u(i,j,k) += 1e-4*F_U0*rnd.random();
                  v(i,j,k) += 1e-4*F_U0*rnd.random();
                  w(i,j,k) += 1e-4*F_U0*rnd.random();
               }
            }
         }
         /* This noise is neither incompressible nor satisfying of the boundary
            conditions.  The code will make it do both after the first timestep.
*/
    

         // Also, write out the  initial velocities and proper M-file readers
         write_reader(u,"u",true);
         write_reader(v,"v",true);
         write_reader(w,"w",true);
         write_array(u,"u",0);
         write_array(v,"v",0);
         write_array(w,"w",0);
         return;
      }

      void init_tracer(int t_num, DTArray & rho) {
         rho = background_strat((*zgrid));
         write_array(rho,"rho",0); write_reader(rho,"rho",true);
      }

      /* Apply forcing in the domain; in this case it is simply due
         to the influcence of gravity on the stratification */
      void forcing(double t, const DTArray & u, DTArray & u_f, 
            const DTArray & v, DTArray & v_f, const DTArray & w,
            DTArray & w_f, vector<DTArray *> & tracers,
            vector<DTArray *> & tracers_f) {
         // Rotation couples u and v, plus a source term for the tide
	u_f = F_U0/F_T0*(t < F_T0);
         v_f = 0;//ROT_F*u;
         w_f = -g*(*tracers[0]);
         // No forcing -on- the density is necessary because it is passive
         *tracers_f[0] = 0;
      }

      /* Basic analysis, to write out the field periodically */
      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> & tracer, DTArray & pressure) {
         /* If it is very close to the plot time, write data fields to disk */
         if ((time - last_plot - plot_interval) > -1e-6) {
            plot_number++;
            if (master()) fprintf(stderr,"*");
            write_array(u,"u",plot_number);
            write_array(v,"v",plot_number);
            write_array(w,"w",plot_number);
            write_array(*tracer[0],"rho",plot_number);
            last_plot = last_plot + plot_interval;
         }
         // Also, calculate and write out useful information: maximum u, w, and t'
         double max_u = psmax(max(abs(u)));
         double max_w = psmax(max(abs(w)));
         double max_t = psmax(max(abs(*tracer[0])));
         if (master()) fprintf(stderr,"%.4f: %.2g %.2g %.2g\n",time,max_u,max_w,max_t);
      }

      mapiw(): hill(split_range(NX))
      { // Initialize the local variables
         plot_number = 0;
         last_plot = 0;
         // Create one-dimensional arrays for the coordinates
      }

};

/* The ``main'' routine */
int main(int argc, char ** argv) {
   /* Initialize MPI.  This is required even for single-processor runs,
      since the inner routines assume some degree of parallelization,
      even if it is trivial. */
   f_order = 4; f_cutoff = 0.8; f_strength = -0.33;
   MPI_Init(&argc, &argv);
   if (argc > 3) { // Check command line arguments
      NX = atoi(argv[1]); // Read in number of horizontal points, if specified
      NY = atoi(argv[2]);
      NZ = atoi(argv[3]); // and vertical
   } 
   if (NX <= 0) {
      NX = 1024;
   }
   if (NY <= 0) {
      NY = 1;
   }
   if (NZ <= 0) {
      NZ = 128;
   }
   if (master()) {
      fprintf(stderr,"Using a grid of %d x %d points\n",NX,NZ);
   }
   mapiw mycode; // Create an instantiated object of the above class
   /// Create a flow-evolver that takes its settings from the above class
   FluidEvolve<mapiw> do_mapiw(&mycode);
   // Run to a final time of 1.
   do_mapiw.initialize();
   do_mapiw.do_run(final_time);
   MPI_Finalize(); // Cleanly exit MPI
   return 0; // End the program
}

