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
int NY = 0;
const double plot_interval = 5; // Time between field writes
const double final_time = 1e-2;

double start_time = 0;
double now;
double then;
int next_bench = 1;
int last_bench = 0;

class helmholtz : public BaseCase {
   public:
      // Variables to set the plot sequence number and time of the last writeout
      Array<double,1> xx, zz;
      int plot_number; double last_plot;
      int itercount;

      // Resolution in X, Y (1), and Z
      int size_x() const { return NX; }
      int size_y() const { return NY; }
      int size_z() const { return NZ; }

      /* Set periodic in x, free slip in z */
      DIMTYPE type_z() const {return FREE_SLIP;}
      DIMTYPE type_default() const { return PERIODIC; }

      /* The grid corresponds to a 1 (x 1) x 1 physical space */
      double length_x() const { return LENGTH_X; }
      double length_y() const { return 1; }
      double length_z() const { return LENGTH_Z; }

      /* Use one actively-modified tracer */
      int numActive() const { return 1; }

      // Use viscosity and diffusivity
      double get_visco() const { return 0; }
      double get_diffusivity(int t_num) const { return 0; }

      /* Start at t=0 */
      double init_time() const { return 0; }

      /* Modify the timestep if necessary in order to land evenly on a plot time */
      double check_timestep (double intime, double now) {
         // Firstly, the buoyancy frequency provides a timescale that is not
         // accounted for with the velocity-based CFL condition.
         //return 1.0/8192;
         return 1e-4;
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
/*         write_reader(u,"u",true);
         write_reader(w,"w",true);
         write_array(u,"u",0);
         write_array(w,"w",0);*/
         return;
      }

      /* Initialze the temperature perturbation to a small value */
      void init_tracer(int t_num, DTArray & rhoprime) {
         /* We want to write out a grid in order to make plots later,
            so let's re-use rhoprime to that end */

         // Assign the x-array to the two-dimensional grid
         rhoprime = xx(ii) + 0*kk;
//         write_array(rhoprime,"xgrid"); write_reader(rhoprime,"xgrid",false);

         // Assign the z-array to the two-dimensional grid
         rhoprime = 0*ii + zz(kk);
//         write_array(rhoprime,"zgrid"); write_reader(rhoprime,"zgrid",false);

         rhoprime = (0.1*cos(pertur_k/2*xx(ii)) + cos(pertur_k*xx(ii)))*
                        pow(cosh((zz(kk)-0.5)/dz_rho),-2)*delta_rho*1e-6;
//         write_array(rhoprime,"rho",0); write_reader(rhoprime,"rho",true);
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
         itercount++;
         if (itercount == 1) {
            now = MPI_Wtime();
            if (master()) fprintf(stderr,"First iteration complete in %gs\n",now-start_time);
            last_bench = itercount;
            next_bench = 10;
         }
         if (itercount == next_bench) {
            // Calculate memory usage from the proc/self/statm file
            int mem_size, mem_res, mem_share;
            // Get the system page size -- often 4KB
            int sys_pagesize = sysconf(_SC_PAGE_SIZE);
            FILE * statm = fopen("/proc/self/statm","r");
            assert(statm);
            // Read in the total memory size, resident set size, and shared size
            fscanf(statm,"%d %d %d",&mem_size,&mem_res,&mem_share);
            fclose(statm);
            // We mostly care about the resident set size, since that represents
            // the true processor load.  Since any individual processor may be
            // unrepresentative on account of MPI-Library weirdness and load
            // balancing, calculate the total and max-per-process memory usage
            double mem_total = pssum(double(mem_res)*sys_pagesize/1024.0); // results in kb
            double mem_max = psmax(double(mem_res)*sys_pagesize/1024.0);
            then = now; now = MPI_Wtime();
            if (master()) fprintf(stderr,
                  "%d iterations complete in %gs (%gs per, %gs marginal)\n" 
                  "       [Mem: total %.2f MB, %.2f MB maximum/proc]\n",
                  itercount,
                  now-start_time, // cumulative time
                  (now-start_time)/itercount, // average time
                  (now-then)/(itercount-last_bench), // average since last writeout
                  mem_total/1024.0, // Total memory (MB)
                  mem_max/1024.0 // max-per-proc (MB)
                  );
            last_bench = itercount;
            next_bench *= 2;
         }
         if ((time - last_plot - plot_interval) > -1e-6) {
            plot_number++;
            if (master()) fprintf(stderr,"*");
            write_array(u,"u",plot_number);
            write_array(w,"w",plot_number);
            write_array(*tracer[0],"rho",plot_number);
            last_plot = last_plot + plot_interval;
         }
         // This code is essentially dead code, but is left in because
         // it accurately represents typical per-timestep analysis;
         // removing it may affect performance results in an uneralistic way.
         double max_u = psmax(max(abs(u)));
         double max_w = psmax(max(abs(w)));
         double max_t = psmax(max(abs(*tracer[0])));
         double usq = pssum(sum(pow(u-0.5*delta_u*tanh((zz(kk)-0.5)/dz_u),2)))/(NX*NZ);
         double wsq = pssum(sum(w*w))/(NX*NZ);
         double rhogh = pssum(sum(*tracer[0]*g*zz(kk)));      
      }

      helmholtz():
         xx(split_range(NX)), zz(NZ), itercount(0)
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
   if (argc > 1) { // Check command line arguments
      NX = atoi(argv[1]); // Read in number of vertical points, if specified
   } 
   if (NX <= 0) {
      NX = 2698;
   }
   if (argc > 2) {
      NZ = atoi(argv[2]);
   } 
   if (NZ <= 0) {
      NZ = 256;
   }
   if (argc > 3) {
      NY = atoi(argv[3]);
   }
   if (NY <= 0) {
      NY = 8;
   }
   int nproc;
   MPI_Comm_size(MPI_COMM_WORLD,&nproc);
   if (master()) {
      fprintf(stderr,"KH billow timings on %d processors with %d x %d x %d grid\n",nproc,NX,NY,NZ);
   }
   helmholtz mycode; // Create an instantiated object of the above class
   /// Create a flow-evolver that takes its settings from the above class
   //zero_tracer_boundary = true;
   start_time = MPI_Wtime();
   FluidEvolve<helmholtz> do_helmholtz(&mycode);
   // Run to a final time of 1.
   do_helmholtz.initialize();
   do_helmholtz.do_run(final_time);
   now = MPI_Wtime();
   if (master()) fprintf(stderr,"Total runtime (%d iterations) complete in %gs (%gs per)\n",mycode.itercount,now-start_time,(now-start_time)/mycode.itercount);
   MPI_Finalize(); // Cleanly exit MPI
   return 0; // End the program
}
