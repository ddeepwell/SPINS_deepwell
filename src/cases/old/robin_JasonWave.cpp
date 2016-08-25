/* A sample case, for illustrating Benard convection cells */

// Required headers 
#include <blitz/array.h> // Blitz++ array library
#include "../TArray.hpp" // Custom extensions to the library to support FFTs
#include "../NSIntegrator.hpp" // Time-integrator for the Navier-Stokes equations
#include <mpi.h> // MPI parallel library
#include "../BaseCase.hpp" // Support file that contains default implementations of several functions


#include <random/normal.h> // Random numbers for initial perturbation

using namespace std;
//using namespace TArrayn;
using namespace NSIntegrator;
using namespace ranlib;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

// Physical constants
const double g     = 9.81;
const double rho_0 = 1; // Units of kg / L
const double alpha = 1; // Thermal expansion of water, in units of kg / L / K
const double kappa = 1e-3; // 1e-3; // Thermal diffusivity in water
const double nu    = 1e-4; //1e-2; // Viscosity of water (divided by rho)


//Density Constants
const double delta_rho = 1; 
const double dz_rho = 0.02; 
const double z0 = 0; 

// Blanchettes Constants
const double R = 1.5; 
const double C0 = 1; // Careful, need to check with intial condition 
const double d  = 100e-6;
const double a  = 1.3e-7;
const double Re_p = d*pow(9.81*d*R,1/2)/(nu); 
const float U_s = 0.002;// //See Matlab Computation     pow(W*9.81*R,1/3)*pow(d,2)/nu; //Settling Velocity  

// Pysical parameters
const double LENGTH_X = 1; // 8cm side
const double LENGTH_Z = 0.25; // 1cm depth
const double DELTA_T = 1.0; // Temperature difference, bottom to top


// Numerical parameters
const int NX = 32;
const int NZ = 32;
const double plot_interval = 0.25; // Time between field writes
const double final_time = 100.0;


class benard : public BaseCase {
   public:
      // Variables to set the plot sequence number and time of the last writeout
      int plot_number; double last_plot;

      const static int T = 0; 
      const static int RHO = 1; 

      // Resolution in X, Y (1), and Z
      int size_x() const { return NX; }
      int size_y() const { return 1; }
      int size_z() const { return NZ; }

      /* Set free-slip in x, Chebyshev in z */
      DIMTYPE type_z() const {return NO_SLIP;}
      //DIMTYPE type_default() const { return FREE_SLIP; }
      DIMTYPE type_default() const { return PERIODIC; }

      // Gradient Operator
      Grad * gradient_op;
      
      //Need some Variable for the shear
      DTArray shear;
      DTArray C_z;
      DTArray Z_B;
      DTArray E_s; 
      DTArray C;

      /* The grid corresponds to a 1 (x 1) x 1 physical space */
      double length_x() const { return LENGTH_X; }
      double length_y() const { return 1; }
      double length_z() const { return LENGTH_Z; }

      /* Use one actively-modified tracer */
      int numActive() const { return 2; }

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

      //No idea what this does by its in doc_diapole.cpp
      /* Record the gradient-taking object. This is given by the NSIntegrator
         Code, and it reflect the boundary types and any Jacobian-transform*/
      void set_grad(Grad  * in_grad){
         gradient_op = in_grad; 
      }
    
      void comp_shear( DTArray  & u, DTArray & v, DTArray & w){
         //Import Grad 
         gradient_op->setup_array(&u,FOURIER,FOURIER,CHEBY);
         //Compute and save u_z 
         gradient_op->get_dz(&shear,false);
         //shear = pow(abs(shear(ii,jj,kk)*nu),0.5); //Might need to change this
      }

      void comp_Cz( vector<DTArray *> & tracers){
         C = *tracers[T];
         //Import Grad 
         gradient_op->setup_array(tracers[T],FOURIER,FOURIER,CHEBY);
         //Compute and save u_z 
         gradient_op->get_dz(&C_z,false);
      }


      /* Initialize velocities at the start of the run.  For this simple
         case, initialize all velocities to 0 */
      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         fprintf(stderr,"U_s = %.4f \n",U_s); 
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

      /* Initialze the temperature perturbation to a small value */
      void init_tracers(vector<DTArray *>  & tracers) {
         fprintf(stderr,"Init_tracer %d\n",2);
        //Initialize Arrays with Tracers
        DTArray & tprime = *tracers[T];
        DTArray & rhop = *tracers[RHO];

         /* We want to write out a grid in order to make plots later,
            so let's re-use tprime to that end */
         // Create one-dimensional arrays for the coordinates
         Array<double,1> xx(split_range(NX)), zz(NZ); 
        Array<double,3> grid(alloc_lbound(NX,1,NZ),
                              alloc_extent(NX,1,NZ),
                              alloc_storage(NX,1,NZ));
         xx = LENGTH_X*(-0.5 + (ii + 0.5)/NX);
         zz = -LENGTH_Z/2*cos(M_PI*ii/(NZ-1));
         
         grid = xx(ii) + 0*jj + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = 0*ii + 0*jj + zz(kk);
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);

         assert (tracers.size() == 2); 
         //tprime = exp((zm - 0.5));
         //tprime = 0*cos((zz(kk) + 0*ii)*3.14159);
         //Initialize t
         tprime = 1e-3;  
         write_array(tprime,"t",0); write_reader(tprime,"t",true);
         
         //initialize Density
         rhop = - 0.05 * (tanh((zz(kk)    - 0.15*exp( -1*pow(xx(ii)/0.1,2))      )/dz_rho) );
         //rhop = -0.1* 0.5*(tanh(((xx(ii) +0.25+ 0*kk)/0.1) ) -tanh((xx(ii)-0.25+0*kk)/0.1));
         //rhop = 0;  
         write_array(rhop,"rho",0); write_reader(rhop,"rho",true);  
      }


      void tracer_bc_z(int t_num, double & dir, double & neu) const {
         // Set up Robin-type BCs
         dir = 0;
         neu = kappa;
      }
      // Forcing in the momentum equations
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
            
         u_f = 0;
         v_f = 0;
         w_f = -g*(*tracers[RHO])/rho_0; 

         // Compute Cz here for use later in tracer_forcing
         comp_Cz(tracers); 
      }
      bool tracer_bc_forcing() const {
         return true;
      }
      // Forcing of the perturbation temperature
      void tracer_forcing(double t, DTArray & u, DTArray & v,
             DTArray & w, vector<DTArray *> & tracers_f) {
        
         //For a Blanchette Based BC we require Uz
         comp_shear(u,v,w); 
         
         Z_B = pow(nu*abs(shear(ii,jj,kk)),0.5)*pow(Re_p,1.23)/U_s*0.586; 
         E_s = (1/C0)*(a*pow(Z_B(ii,jj,kk),5))/(1+(a/0.3)*pow(Z_B(ii,jj,kk),5));  
         //fprintf(stderr,"Shear = %.4f %.4f\n",shear(0,0,0),shear(0,0,NZ-1) );
         //fprintf(stderr,"Z_B = %.4f %.4f\n",Z_B(0,0,0),Z_B(0,0,NZ-1) );
         //fprintf(stderr,"E_s = %f\n",E_s(0,0,0)); 
         //fprintf(stderr,"Top = %.4f, Bottom = %.4f\n",zm(0,0,NZ-1),zm(0,0,0) );
        
   
         //Force the Sediment
         *tracers_f[T] =  U_s*C_z; //We just want u_z
         // Heat does the flows in at the bottom
         (*tracers_f[T])(blitz::Range::all(),blitz::Range::all(),0) = 
            U_s*E_s(blitz::Range::all(), blitz::Range::all(),0)
            + (-U_s*C(blitz::Range::all(), blitz::Range::all(),0));
         //Out of the top
         (*tracers_f[T])(blitz::Range::all(),blitz::Range::all(),NZ-1)  = 0; //Neumann

         
         //Force The Density
         (*tracers_f[RHO]) = 0; 

      }


      /* The analysis routines are called at each timestep, since it's
         impossible to predict in advance just what will be interesting.  For
         now, this function will do nothing. */
      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer, DTArray & pressure) {
//            vector<DTArray *> & tracer, DTArray & pressure) {
         /* If it is very close to the plot time, write data fields to disk */
         if ((time - last_plot - plot_interval) > -1e-6) {
            plot_number++;
            if (master()) fprintf(stderr,"*");
            write_array(u,"u",plot_number);
            write_array(w,"w",plot_number);
            write_array(*tracer[T],"t",plot_number);
            write_array(*tracer[RHO],"rho",plot_number); 
            last_plot = last_plot + plot_interval;
         }
         // Also, calculate and write out useful information: maximum u, w, and t'
         double max_u = psmax(max(abs(u)));
         double max_w = psmax(max(abs(w)));
         double max_t = psmax(max(abs(*tracer[T])));
         double max_ES  = max(abs(E_s(blitz::Range::all(),0,0)));
         double max_rho = psmax(max(abs(*tracer[RHO]))); 
         if (master()) fprintf(stderr,"%.4f: %.4g %.4g || %.4g %.4g || %.4g\n",time,max_u,max_w,max_t,max_rho,max_ES);
      }

      benard():
         shear(alloc_lbound(NX,1,NZ),alloc_extent(NX,1,NZ),alloc_storage(NX,1,NZ)),
         Z_B(alloc_lbound(NX,1,NZ),alloc_extent(NX,1,NZ),alloc_storage(NX,1,NZ)),
         E_s(alloc_lbound(NX,1,NZ),alloc_extent(NX,1,NZ),alloc_storage(NX,1,NZ)),
         C_z(alloc_lbound(NX,1,NZ),alloc_extent(NX,1,NZ),alloc_storage(NX,1,NZ)),
         C(alloc_lbound(NX,1,NZ),alloc_extent(NX,1,NZ),alloc_storage(NX,1,NZ))
      { // Initialize the local variables
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
   FluidEvolve<BaseCase> do_benard(&mycode);
   // Run to a final time of 1.
   do_benard.initialize();
   do_benard.do_run(final_time);
   MPI_Finalize(); // Cleanly exit MPI
   return 0; // End the program
}

