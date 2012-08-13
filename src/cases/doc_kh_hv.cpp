/* A sample case, for illustrating Benard convection cells */

// Required headers 
#include <blitz/array.h> // Blitz++ array library
#include "../TArray.hpp" // Custom extensions to the library to support FFTs
#include "../NSIntegrator.hpp" // Time-integrator for the Navier-Stokes equations
#include <mpi.h> // MPI parallel library
#include "../BaseCase.hpp" // Support file that contains default implementations of several functions

using namespace std;
using namespace NSIntegrator;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

// Physical constants
const double g = 9.81;
const double rho_0 = 1; // Units of kg / L

// Pysical parameters
const double pertur_k = 2.38434; // Wavelength of most unstable perturbation
const double LENGTH_X = 8*M_PI/pertur_k; // 4 times the most unstable wavelength
const double LENGTH_Z = 1; // depth 1
const double delta_rho = 0.01; // Top to bottom density difference
const double RI = 0.15; // Richardson number at the centre of the picnocline
const double dz_rho = 0.1; // Transition length for rho
const double dz_u = 0.1; // Transition length for u

const double N2_max = g*delta_rho/2/dz_rho; // Maximum N2
const double delta_u = 2*dz_u*sqrt(N2_max/RI); // Top-to-bottom shear

// Numerical parameters
int NZ = 0; // Number of vertical points.  Number of horizontal points
int NX = 0; // will be calculated based on this.
const double plot_interval = 5; // Time between field writes
const double final_time = 50.0;


class helmholtz : public BaseCase {
   public:
      // Variables to set the plot sequence number and time of the last writeout
      Array<double,1> xx, zz;
      int plot_number; double last_plot;

      // Resolution in X, Y (1), and Z
      int size_x() const { return NX; }
      int size_y() const { return 1; }
      int size_z() const { return NZ; }

      /* Set periodic in x, free slip in z */
      DIMTYPE type_z() const {return FREE_SLIP;}
      DIMTYPE type_default() const { return PERIODIC; }

      /* The grid size is governed through the #defines above */
      double length_x() const { return LENGTH_X; }
      double length_y() const { return 1; }
      double length_z() const { return LENGTH_Z; }

      /* Use one actively-modified tracer */
      int numActive() const { return 1; }

      // Use 0 viscosity and diffusivity
      double get_visco() const { return 0; }
      double get_diffusivity(int t_num) const { return 0; }

      /* Start at t=0 */
      double init_time() const { return 0; }

      /* Modify the timestep if necessary in order to land evenly on a plot time */
      double check_timestep (double intime, double now) {
         // Firstly, the buoyancy frequency provides a timescale that is not
         // accounted for with the velocity-based CFL condition.
         return 1e-3;
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
         
         u = 0.5*delta_u*tanh((zz(kk)-0.5)/dz_u); // Use the Blitz++ syntax for simple initialization
         v = 0; // of an entire (2D or 3D) array with a single line
         w = 0; // of code.
         // Also, write out the (zero) initial velocities and proper M-file readers
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

         rhoprime = (cos(pertur_k*xx(ii)))* pow(cosh((zz(kk)-0.5)/dz_rho),-2)*
                     delta_rho*1e-1;
         write_array(rhoprime,"rho",0); write_reader(rhoprime,"rho",true);
      }

      // Forcing in the momentum equations
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         u_f = 0; v_f = 0;
         w_f = -g*((*tracers[0]))/rho_0;
      }
      // Forcing of the perturbation temperature
      void tracer_forcing(double t, DTArray & u, DTArray & v,
            DTArray & w, vector<DTArray *> & tracers_f) {
         /* Since the perturbation temperature is a perturbation, its forcing is
            proportional to the background temperature gradient and the w-velocity */
         *tracers_f[0] = w(ii,jj,kk)*0.5*delta_rho/dz_rho*pow(cosh((zz(kk)-0.5)/dz_rho),-2);
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
            write_array(*tracer[0],"rho",plot_number);
            last_plot = last_plot + plot_interval;
         }
         // Also, calculate and write out useful information: maximum u, w, and t'
         double max_u = psmax(max(abs(u)));
         double max_w = psmax(max(abs(w)));
         double max_t = psmax(max(abs(*tracer[0])));
         // Energetics: mean(u^2), mean(w^2), and mean(rho*h)
         double usq = pssum(sum(u*u))/(NX*NZ);
         double wsq = pssum(sum(w*w))/(NX*NZ);
         double rhogh = pssum(sum(*tracer[0]*g*zz(kk)));
         if (master()) fprintf(stderr,"%.2f: %.2g %.2g %.2g\n",time,max_u,max_w,max_t);
         if (master()) {
            FILE * vels_output = fopen("velocity_output.txt","a");
            if (vels_output == 0) {
               fprintf(stderr,"Unable to open velocity_output.txt for writing\n");
               exit(1);
            }
            fprintf(vels_output,"%.16g %.16g %.16g %.16g %.16g %.16g %.16g\n",
                  time,max_u,max_w,max_t,usq,wsq,rhogh);
            fclose(vels_output);
         }
      }

      helmholtz():
         xx(split_range(NX)), zz(NZ)
      { // Initialize the local variables
         plot_number = 0;
         last_plot = 0;
         // Create one-dimensional arrays for the coordinates
         xx = LENGTH_X*(-0.5 + (ii + 0.5)/NX);
         zz = LENGTH_Z*((ii+0.5)/NZ);
      }

};

/* The ``main'' routine */
int main(int argc, char ** argv) {
   /* Initialize MPI.  This is required even for single-processor runs,
      since the inner routines assume some degree of parallelization,
      even if it is trivial. */
   MPI_Init(&argc, &argv);
   f_strength = -.25;
   f_order = 4;
   f_cutoff = 0.8;
   if (argc > 1) { // Check command line arguments
      NZ = atoi(argv[1]); // Read in number of vertical points, if specified
   } else {
      NZ = 256;
   }
   NX = rint(NZ*LENGTH_X/LENGTH_Z);
   if (master()) {
      fprintf(stderr,"Using a grid of %d x %d points\n",NX,NZ);
   }
   helmholtz mycode; // Create an instantiated object of the above class
   /// Create a flow-evolver that takes its settings from the above class
   FluidEvolve<helmholtz> do_helmholtz(&mycode);
   // Run to a final time of 1.
   do_helmholtz.initialize();
   do_helmholtz.do_run(final_time);
   MPI_Finalize(); // Cleanly exit MPI
   return 0; // End the program
}

