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
const double rho_0 = 1028; // Units of kg / m^3

// Pysical parameters
const double ROT_F = 0.5e-4; // 1/s, mid-latitude
const double LENGTH_X = 4e5; // 1000km
const double LENGTH_Z = 5000; // Water depth
const double N2_max = 1e-3; // Linear stratification buoyancy frequency

const double S0 = 35; // Baseline salinity
const double beta = 7.6e-4; // Density change from salt content

// Hill parameters
const double H_HEIGHT = 1500; // Hill height
const double H_LENGTH = 12e3; // Hill length

// Tide parameters
const double TIDE_PERIOD = 44712;
const double TIDE_M2 = 2*M_PI/TIDE_PERIOD; // M2 tidal frequency
const double TIDE_STRENGTH = 0.01; // Desired maximum tidal current

// Numerical parameters
int NZ = 0; // Number of vertical points.  Number of horizontal points
int NX = 0; // will be calculated based on this.
const double plot_interval = TIDE_PERIOD/32; // Time between field writes
const double final_time = 4*TIDE_PERIOD;


class mapiw : public BaseCase {
   public:
      // Variables to set the plot sequence number and time of the last writeout
      DTArray * xgrid, * zgrid;
      Array<double,1> hill;
      int plot_number; double last_plot;

      // Resolution in X, Y (1), and Z
      int size_x() const { return NX; }
      int size_y() const { return 1; }
      int size_z() const { return NZ; }

      /* Set periodic in x, free slip in z */
      DIMTYPE type_z() const {return NO_SLIP;}
      DIMTYPE type_default() const { return PERIODIC; }

      /* The grid size is governed through the #defines above */
      double length_x() const { return LENGTH_X; }
      double length_y() const { return 1; }
      double length_z() const { return LENGTH_Z; }

      bool is_mapped() const {return true;}
      void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
         xgrid = alloc_array(NX,1,NZ);
         zgrid = alloc_array(NX,1,NZ);
         Array<double,1> xx(split_range(NX)), zz(NZ);
         // Use periodic coordinates in horizontal
         xx = LENGTH_X*(ii+0.5)/NX; // x-coordinate
         zz = cos(ii*M_PI/(NZ-1)); // Chebyshev in vertical

         xg = xx(ii) + 0*jj + 0*kk;
         *xgrid = xg;

         hill = H_HEIGHT*exp(-pow((xx(ii)-LENGTH_X/2)/H_LENGTH,2));
         zg = -LENGTH_Z/2+LENGTH_Z/2*zz(kk) + 0.5*(1-zz(kk))*
                  hill(ii);;
         *zgrid = zg;

         yg = 0;

         write_array(xg,"xgrid");
         write_reader(xg,"xgrid",false);
         write_array(zg,"zgrid");
         write_reader(zg,"zgrid",false);
      }
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
         u = 0;
         v = -TIDE_STRENGTH*ROT_F/TIDE_M2*LENGTH_Z/(LENGTH_Z-hill(ii));
         w = 0;
         // Also, write out the (zero) initial velocities and proper M-file readers
         write_reader(u,"u",true);
         write_reader(v,"v",true);
         write_reader(w,"w",true);
         write_array(u,"u",0);
         write_array(v,"v",0);
         write_array(w,"w",0);
         return;
      }

      void init_tracer(int t_num, DTArray & salt) {
         // The primary constituent of density is salt, so that is 
         // initialize here

         salt = S0 + (N2_max*N2_max)/(-beta*g)*(*zgrid)(ii,jj,kk);
         write_array(salt,"s",0); write_reader(salt,"s",true);
      }

      // Forcing must be done generally, since both rotation and density are
      // involved
      void forcing(double t, DTArray & u, DTArray & u_f, 
            DTArray & v, DTArray & v_f, DTArray & w,
            DTArray & w_f, vector<DTArray *> & tracers,
            vector<DTArray *> & tracers_f) {
         // Rotation couples u and v, plus a source term for the tide
         u_f = -ROT_F*v + cos(t*TIDE_M2)*TIDE_STRENGTH*
            (TIDE_M2-ROT_F*ROT_F/TIDE_M2*LENGTH_Z/(LENGTH_Z-hill(ii)));
         v_f = ROT_F*u;
         w_f = -g*beta*(*tracers[0]-S0);
         // And since the salt content is expressed as total content rather
         // than perturbation, no forcing is necessary.
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
            write_array(*tracer[0],"s",plot_number);
            last_plot = last_plot + plot_interval;
         }
         // Also, calculate and write out useful information: maximum u, w, and t'
         double max_u = psmax(max(abs(u)));
         double max_w = psmax(max(abs(w)));
         double max_t = psmax(max(abs(*tracer[0])));
         if (master()) fprintf(stderr,"%.2f: %.2g %.2g %.2g\n",time,max_u,max_w,max_t);
//         if (master()) {
//            FILE * vels_output = fopen("velocity_output.txt","a");
//            if (vels_output == 0) {
//               fprintf(stderr,"Unable to open velocity_output.txt for writing\n");
//               exit(1);
//            }
//            fprintf(vels_output,"%.16g %.16g %.16g %.16g %.16g %.16g %.16g\n",
//                  time,max_u,max_w,max_t,usq,wsq,rhogh);
//            fclose(vels_output);
//         }
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
   MPI_Init(&argc, &argv);
   if (argc > 2) { // Check command line arguments
      NX = atoi(argv[1]); // Read in number of horizontal points, if specified
      NZ = atoi(argv[2]); // and vertical
   } 
   if (NX <= 0) {
      NX = 2048;
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

