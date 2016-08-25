/* A tutorial case for a periodic, no-slip dam-break problem, where
   fluids of two densities are initially separated by a vertical
   interface */

// Required headers 
#include <blitz/array.h> // Blitz++ array library
#include "../TArray.hpp" // Custom extensions to the library to support FFTs
#include "../NSIntegrator.hpp" // Time-integrator for the Navier-Stokes equations
#include <mpi.h> // MPI parallel library
#include "../BaseCase.hpp" // Support file that contains default implementations of several functions
#include <random/normal.h> // Blitz random number generator

using namespace std;
using namespace NSIntegrator;
using namespace ranlib;


// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

// Physical constants
const double g = 9.81;
const double rho_0 = 1; // Units of kg / L

// Grid scales

const double LENGTH_X = 10; // meters
const double LENGTH_Z = 1;
const double LENGTH_Y = 1;

// Stratification parameters
const double delta_rho = 0.01; // ± difference from rho_0
const double delta_x = 0.25; // Transition length

const double clap_on = 2.5; // Turn-on location for high-density fluid
const double clap_off = 7.5; // Turn-off "

// Viscosity and diffusivity
const double VISCO = 3e-5;
const double DIFFU = 3e-5;

// Numerical parameters
const int NX = 2048; // Points in x-direction
const int NZ = 128; // z-direction
const int NY = 1; // spanwise

const double plot_interval = 1; // Time between field writes
const double final_time = 200.0;

/* Derived parameters */

// Fraction of domain filled by the heavier fluid
const double filled_frac = (clap_off-clap_on)/LENGTH_X;

// Phase speed of long waves if the stratification were vertical
// rather than horizontal
const double c0 = sqrt(g*2*delta_rho/rho_0*(filled_frac)*(1-filled_frac)*LENGTH_Z);

// Maximum buoyancy frequency (squared) if the initial 
// stratification was stable
const double N2_max = 2*delta_rho/rho_0*g/delta_x;

// Reynolds number based on this phase speed and domain depth
const double Re = c0*LENGTH_Z/VISCO;

class dambreak : public BaseCase {
   public:
      // Arrays for 1D grids defined here
      Array<double,1> xx, zz;

      // Helper variables for the plot number and time of
      // last plotting
      int plot_number; double last_plot;

      // Resolution in X, Y, and Z
      int size_x() const { return NX; }
      int size_y() const { return NY; }
      int size_z() const { return NZ; }

      /* Set the z-dimension to be no-slip, with periodic
         expansions in x and y */
      DIMTYPE type_z() const {return NO_SLIP;}
      DIMTYPE type_default() const { return PERIODIC; }

      /* The grid size is governed through the #defines above */
      double length_x() const { return LENGTH_X; }
      double length_y() const { return LENGTH_Y; }
      double length_z() const { return LENGTH_Z; }

      /* Use one actively-modified tracer */
      int numActive() const { return 1; }

      double get_visco() const { return VISCO; }
      double get_diffusivity(int t_num) const { return DIFFU; }

      /* Start at t=0 */
      double init_time() const { return 0; }

      /* Modify the timestep if necessary in order to land evenly on a plot time */
      double check_timestep (double intime, double now) {
         // Firstly, the buoyancy frequency provides a timescale that is not
         // accounted for with the velocity-based CFL condition.
         if (intime > 0.5/sqrt(N2_max)) {
            intime = 0.5/sqrt(N2_max);
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
         case, initialize all velocities to 0 */
      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         u = 0; // Use the Blitz++ syntax for simple initialization
         v = 0; // of an entire (2D or 3D) array with a single line
         w = 0; // of code.
         // Also, write out the (zero) initial velocities and proper M-file readers
         /* Add random initial perturbation */
         int myrank;
         MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
         /* Add random noise about 3 orders of magnitude below dipole */
         Normal<double> rnd(0,1);
         rnd.seed(myrank);
         for (int i = u.lbound(firstDim); i<= u.ubound(firstDim); i++) {
            for (int j = u.lbound(secondDim); j<= u.ubound(secondDim); j++) {
               for (int k = u.lbound(thirdDim); k<= u.ubound(thirdDim); k++) {
                  u(i,j,k) += 1e-3*c0*rnd.random();
                  //v(i,j,k) += 1e-2*rnd.random();
                  w(i,j,k) += 1e-3*c0*rnd.random();
               }
            }
         }
         write_reader(u,"u",true);
         write_reader(w,"w",true);
         write_array(u,"u",0);
         write_array(w,"w",0);
         return;
      }

      /* Initialze the temperature perturbation to a small value */
      void init_tracer(int t_num, DTArray & rhoprime) {
         
         /* We want to write out a grid in order to make plots later,
            so let's re-use rhoprime to that end */

         // Assign the x-array to the two-dimensional grid
         rhoprime = xx(ii) + 0*kk;
         write_array(rhoprime,"xgrid"); write_reader(rhoprime,"xgrid",false);

         // Assign the z-array to the two-dimensional grid
         rhoprime = 0*ii + zz(kk);
         write_array(rhoprime,"zgrid"); write_reader(rhoprime,"zgrid",false);

         rhoprime = delta_rho*(tanh((xx(ii)-clap_on)/delta_x) - tanh((xx(ii)-clap_off)/delta_x)-1);
         write_array(rhoprime,"rho",0); write_reader(rhoprime,"rho",true);
      }

      // Forcing in the momentum equations
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         u_f = 0; v_f = 0;
         w_f = -g*((*tracers[0]))/rho_0;
      }
      // Forcing of the density (zero)
      void tracer_forcing(double t, DTArray & u, DTArray & v,
            DTArray & w, vector<DTArray *> & tracers_f) {
         *tracers_f[0] = 0;
      }

      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> & tracer, DTArray & pressure) {
         /* If it is very close to the plot time, write data fields to disk */
         if ((time - last_plot - plot_interval) > -1e-6) {
            plot_number++;
            if (master()) fprintf(stderr,"*");
            write_array(u,"u",plot_number);
            write_array(w,"w",plot_number);
            write_array(*tracer[0],"rho",plot_number);
            last_plot = last_plot + plot_interval;
         }
         // Also, calculate and write out useful information: maximum u, w, and t'
         double max_u = psmax(max(abs(u)));
         double max_w = psmax(max(abs(w)));
         double max_t = psmax(max(abs(*tracer[0])));
         if (master()) fprintf(stderr,"%.2f: %.2g %.2g %.2g\n",time,max_u,max_w,max_t);
      }

      // Constructor 
      dambreak(): // Initialization list for xx/zz 1d grids
         xx(split_range(NX)), zz(NZ)
      { // Initialize the local variables
         plot_number = 0;
         last_plot = 0;
         // Create one-dimensional arrays for the coordinates
         xx = LENGTH_X*((ii + 0.5)/NX);
         zz = LENGTH_Z*(0.5-0.5*cos(M_PI*ii/(NZ-1)));
      }

};

/* The ``main'' routine */
int main(int argc, char ** argv) {
   /* Initialize MPI.  This is required even for single-processor runs,
      since the inner routines assume some degree of parallelization,
      even if it is trivial. */
   MPI_Init(&argc, &argv);
   if (master()) {
      fprintf(stderr,"Dam break problem\n");
      fprintf(stderr,"Using a %f x %f x %f grid of %d x %d x %d points\n",LENGTH_X,LENGTH_Y,LENGTH_Z,NX,NY,NZ);
      fprintf(stderr,"g = %f, rho_0 = %f, delta_rho %f\n",g,rho_0,delta_rho);
      fprintf(stderr,"Stably-stratified phase speed %g\n",c0);
      fprintf(stderr,"Reynolds number %g\n",Re);
   }
   dambreak mycode; // Create an instantiated object of the above class
   /// Create a flow-evolver that takes its settings from the above class
   FluidEvolve<dambreak> do_stuff(&mycode);

   // Initialize
   do_stuff.initialize();

   // Run until the end of time
   do_stuff.do_run(final_time);
   MPI_Finalize(); // Cleanly exit MPI
   return 0; // End the program
}

