/* A sample case, for illustrating Benard convection cells */

// Required headers 
#include <blitz/array.h> // Blitz++ array library
#include "../TArray.hpp" // Custom extensions to the library to support FFTs
#include "../NSIntegrator.hpp" // Time-integrator for the Navier-Stokes equations
#include <mpi.h> // MPI parallel library
#include "../BaseCase.hpp" // Support file that contains default implementations of several functions

#include <random/normal.h> // Random numbers for initial perturbation

using namespace std;
using namespace NSIntegrator;
using namespace ranlib;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

// Physical constants
const double g = 1;
const double rho_0 = 1; // Units of kg / L
const double alpha = 1; // Thermal expansion of water, in units of kg / L / K
const double kappa = 1e-5; // Thermal diffusivity in water
const double nu = 1e-4; // Viscosity of water (divided by rho)

// Pysical parameters
const double LENGTH_X = 1; // 8cm side
const double LENGTH_Z = 1; // 1cm depth
const double DELTA_T = 1.0; // Temperature difference, bottom to top

// Numerical parameters
const int NX = 512;
const int NZ = 256;
const double plot_interval = 0.25; // Time between field writes
const double final_time = 400.0;


class benard : public BaseCase {
   public:
      // Variables to set the plot sequence number and time of the last writeout
      int plot_number; double last_plot;

      // Resolution in X, Y (1), and Z
      int size_x() const { return NX; }
      int size_y() const { return 1; }
      int size_z() const { return NZ; }

      /* Set free-slip in x, Chebyshev in z */
      DIMTYPE type_z() const {return NO_SLIP;}
      DIMTYPE type_default() const { return FREE_SLIP; }

      /* The grid corresponds to a 1 (x 1) x 1 physical space */
      double length_x() const { return LENGTH_X; }
      double length_y() const { return 1; }
      double length_z() const { return LENGTH_Z; }

      /* Use one actively-modified tracer */
      int numActive() const { return 1; }

      // Use viscosity and diffusivity
      double get_visco() const { return nu; }
      double get_diffusivity(int t_num) const { return kappa; }

      /* Start at t=0 */
      double init_time() const { return 0; }

      /* Modify the timestep if necessary in order to land evenly on a plot time */
      double check_timestep (double intime, double now) {
         // Firstly, the buoyancy frequency provides a timescale that is not
         // accounted for with the velocity-based CFL condition.
         if (intime > 1e-2) {
            intime = 1e-2;
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
         fprintf(stderr,"Init_vels\n");
         u = 0; // Use the Blitz++ syntax for simple initialization
         v = 0; // of an entire (2D or 3D) array with a single line
         w = 0; // of code.
         // Also, write out the (zero) initial velocities and proper M-file readers
         write_reader(u,"u",true);
         write_reader(w,"w",true);
         write_array(u,"u",0);
         write_array(w,"w",0);
         return;
      }

      void tracer_bc_z(int t_num, double & dir, double & neu) const {
         // Set up Robin-type BCs
         dir = 1;
         neu = 0.1;
      }
      /* Initialze the temperature perturbation to a small value */
      void init_tracer(int t_num, DTArray & tprime) {
         fprintf(stderr,"Init_tracer %d\n",t_num);
         /* We want to write out a grid in order to make plots later,
            so let's re-use tprime to that end */
         // Create one-dimensional arrays for the coordinates
         Array<double,1> xx(split_range(NX)), zz(NZ);
         xx = LENGTH_X*(-0.5 + (ii + 0.5)/NX);
         zz = -LENGTH_Z/2*cos(M_PI*ii/(NZ-1));
         //zz = LENGTH_Z*(-1 + (ii+0.5)/NZ);

         // Assign the x-array to the two-dimensional grid
         tprime = xx(ii) + 0*kk;
         write_array(tprime,"xgrid"); write_reader(tprime,"xgrid",false);

         // Assign the z-array to the two-dimensional grid
         tprime = 0*ii + zz(kk);
         write_array(tprime,"zgrid"); write_reader(tprime,"zgrid",false);

         /* Now, create a small perturbation temperature to trigger convective
            flow and any instabilities */
         Normal<double> rnd(0,1);  // Normal random variable of variance 1
         for (int i = tprime.lbound(firstDim); i <= tprime.ubound(firstDim); i++) {
            /* Re-seed the random number generator with each row.  This ensures that
               the same random perturbation will be used regardless of how many 
               processors the program runs on.  For a particular grid resolution,
               it also makes the perturbation be the same from run to run.  This is
               useful for debugging, but needs to be changed in order to compute any
               ensemble statistic.s */
            rnd.seed(i);
            for (int j = tprime.lbound(thirdDim); j <= tprime.ubound(thirdDim); j++) {
               // Apply a small perturbation, normalized by the temperature scale
               // and cell area.
               tprime(i,0,j) = DELTA_T + rnd.random()*DELTA_T*1e-3;
            }
         }
         write_array(tprime,"t",0); write_reader(tprime,"t",true);
      }

      // Forcing in the momentum equations
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         u_f = 0; v_f = 0;
         w_f = g*(alpha*(*tracers[0]))/rho_0;
      }
      // Forcing of the perturbation temperature
      void tracer_forcing(double t, DTArray & u, DTArray & v,
            DTArray & w, vector<DTArray *> & tracers_f) {
         *tracers_f[0] = 0;
      }


      /* The analysis routines are called at each timestep, since it's
         impossible to predict in advance just what will be interesting.  For
         now, this function will do nothing. */
      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> & tracer, DTArray & pressure) {
         /* If it is very close to the plot time, write data fields to disk */
         if ((time - last_plot - plot_interval) > -1e-6) {
            plot_number++;
            if (master()) fprintf(stderr,"*");
            write_array(u,"u",plot_number);
            write_array(w,"w",plot_number);
            write_array(*tracer[0],"t",plot_number);
            last_plot = last_plot + plot_interval;
         }
         // Also, calculate and write out useful information: maximum u, w, and t'
         double max_u = psmax(max(abs(u)));
         double max_w = psmax(max(abs(w)));
         double max_t = psmax(max(abs(*tracer[0])));
         if (master()) fprintf(stderr,"%.4f: %.4g %.4g %.4g\n",time,max_u,max_w,max_t);
      }

      benard() { // Initialize the local variables
         plot_number = 0;
         last_plot = 0;
      }

};

/* The ``main'' routine */
int main(int argc, char ** argv) {
   /* Initialize MPI.  This is required even for single-processor runs,
      since the inner routines assume some degree of parallelization,
      even if it is trivial. */
   MPI_Init(&argc, &argv);
   benard mycode; // Create an instantiated object of the above class
   /// Create a flow-evolver that takes its settings from the above class
   FluidEvolve<benard> do_benard(&mycode);
   // Run to a final time of 1.
   do_benard.initialize();
   do_benard.do_run(final_time);
   MPI_Finalize(); // Cleanly exit MPI
   return 0; // End the program
}

