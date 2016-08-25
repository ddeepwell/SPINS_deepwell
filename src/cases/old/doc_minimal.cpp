/* A minimal example case for the fluids model.  When compiled and run, this
   exits cleanly after performing no useful computation.  However, this also
   shows the basic required structure for any (useful) case for this
   underlying model. */

// Required headers 
#include <blitz/array.h> // Blitz++ array library
#include "../TArray.hpp" // Custom extensions to the library to support FFTs
#include "../NSIntegrator.hpp" // Time-integrator for the Navier-Stokes equations
#include <mpi.h> // MPI parallel library
#include "../BaseCase.hpp" // Support file that contains default implementations of several functions

using namespace std;
using namespace NSIntegrator;

class minimal : public BaseCase {
   public:
      /* Set up a 100 x 1 x 100 grid */
      int size_x() const { return 100; }
      int size_y() const { return 1; }
      int size_z() const { return 100; }

      /* Set all boundaries to be periodic */
      DIMTYPE type_default() const { return PERIODIC; }

      /* The grid corresponds to a 1 (x 1) x 1 physical space */
      double length_x() const { return 1; }
      double length_y() const { return 1; }
      double length_z() const { return 1; }

      /* Use no tracer variables */
      int numtracers() const { return 0; }

      /* Start at t=0 */
      double init_time() const { return 0; }

      /* Initialize velocities at the start of the run.  For this simple
         case, initialize all velocities to 0 */
      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         u = 0; // Use the Blitz++ syntax for simple initialization
         v = 0; // of an entire (2D or 3D) array with a single line
         w = 0; // of code.
         return;
      }

      /* The analysis routines are called at each timestep, since it's
         impossible to predict in advance just what will be interesting.  For
         now, this function will do nothing. */
      void vel_analysis(double t, DTArray & u, DTArray & v, DTArray & w) {
         return;
      }
};

/* The ``main'' routine */
int main(int argc, char ** argv) {
   /* Initialize MPI.  This is required even for single-processor runs,
      since the inner routines assume some degree of parallelization,
      even if it is trivial. */
   MPI_Init(&argc, &argv);
   minimal mycode; // Create an instantiated object of the above class
   /// Create a flow-evolver that takes its settings from the above class
   FluidEvolve<minimal> do_nothing(&mycode);
   // Initialize the flow
   do_nothing.initialize();
   // Run to a final time of 1.
   do_nothing.do_run(1);
   MPI_Finalize(); // Cleanly exit MPI
   return 0; // End the program
}

