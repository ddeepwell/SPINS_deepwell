/* Barotropic instability of a jet on mesoscales, with non-hydrostatic equations */

#include "../Science.hpp"
#include "../TArray.hpp"
#include "../Par_util.hpp"
#include "../NSIntegrator.hpp"
#include "../BaseCase.hpp"
#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <random/normal.h>

/* Dimensional and non-dimensional parameters */

/* As a note, contrary to usual practice, u and -w- are horizontal
   velocities, and v is a vertical velocity.  This problem has a primarily
   2D instability governed by rotation, and the horizontal dimensions are
   most important for near-geostrophic parameters. */

#define LSCALE 1e4
#define Lx (LSCALE*20 )
#define Lz (LSCALE*10)
#define Ly 1e3 
#define DELTA (Ly/LSCALE) // Aspect ratio

#define MIN_X (-Lx/2)
#define MIN_Y (-Ly/2)
#define MIN_Z (-Lz/2)

#define ROT_F 1e-4 // traditional Coriolis parameter
#define ROT_B 1e-4 //1e-4 // Nontraditional Coriolis parameter

#define ROSSBY 1.0 // Rossby Number
#define FROUDE_INV 1.0 // Inverse of Froude number (so no stratification is 0)

#define U_SCALE (ROSSBY*ROT_F*LSCALE) // Scale of U
#define BUOY_SCALE (U_SCALE*FROUDE_INV/Ly) // Scale of N^2

// And lastly, the jet width.  The nondimensionalization is on
// the grid scale, so this parameter is not reflected in the above

#define JET_WIDTH (LSCALE)

// Numerical parameters

#define Nx 640  // Number of points in x
#define Nz 320  // in z
#define Ny 32 // One point for strictly 2D run

using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

using namespace ranlib;

/* Rank is used as a seed for the random noise at initializaition */
int myrank = -1;

/* Variables to allow restarting */
bool restarting = false;
int restart_sequence = 0;
double restart_time = 0;

/* Various logs to output */
double times_log[100], // Timesteps taken
       ke_2d_log[100], ke_3d_log[100], // 2 and 3D kinetic energies
       ke_p_log[100], // Perturbation KE
       max_v_log[100], // Maximum v
       max_rho_log[100]; // Maximum rho_prime

class userControl : public BaseCase {
   public:
      int plotnum, itercount, lastplot, last_writeout;
      bool plot_now;
      double plot_interval, nextplot;
      double start_rho_spread;

      Array<double,1> xx,yy,zz;
      Array<double,1> base_rho;
      Array<double,1> phase_y;

      int size_x() const { return Nx; }
      int size_y() const { return Ny; }
      int size_z() const { return Nz; }

      /* Define a channel periodic in x, with free slip walls
         (in z) and free-slip walls in depth (y) */
      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_y() const { return FREE_SLIP; }
      DIMTYPE type_z() const { return FREE_SLIP; }

      double length_x() const { return Lx; }
      double length_y() const { return Ly; }
      double length_z() const { return Lz; }

      /* Turn off viscosity and diffusivity */
      double get_visco() const {
         return 0; 
      }
      double get_diffusivity(int t) const {
         return 0; 
      }

      double init_time() const {
         return restart_time;
      }

      /* We have an active tracer, namely density */
      int numActive() const { return 1; }

      /* Timestep-check function.  This (long with "write everything" outputs) should
         really be bumped into the BaseCase */
      double check_timestep(double intime, double now) {
//         if (master()) fprintf(stderr,"Input time %g\n",intime);
         if (intime < 1e-9) {
            /* Timestep's too small, somehow stuff is blowing up */
            if (master()) fprintf(stderr,"Tiny timestep, aborting\n");
            return -1;
         } else if (size_y() > 1 && 1/intime < 2*sqrt(BUOY_SCALE)) {
            /* The bouyancy frequency provides another fast timescale.  The
               bouyancy period is 1/sqrt(BUOY_SCALE), so if intime > 0.25/BUOY_SCALE
               we want to clamp it down.  Since a nonstratified problem has
               BUOY_SCALE = 0, instead look at 1/intime < 4*BUOY_SCALE */
            intime = 0.5/sqrt(BUOY_SCALE);
         }
        if (intime > 0.25/ROT_F) { // rotation timescale
           intime = 0.25/ROT_F;
        }
         /* Calculate how long we have until the next plottime, and then adjust
            the timestep such that we take a whole number of steps to ge there */
         double until_plot = nextplot - now;
         double steps = ceil(until_plot / intime);
         double real_until_plot = steps*intime;

         if (fabs(until_plot - real_until_plot) < 1e-5*plot_interval) {
            /* Close enough for scientific work */
            return intime;
         } else {
            /* Adjust the timestep */
            return (until_plot / steps);
         }
      }
      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer, DTArray & pressure) {
         itercount = itercount + 1;
         double mddiff, ke_2d, ke_3d, max_v, ke_p;

         mddiff = pvmax(*tracer[0]) - pvmin(*tracer[0]);
         ke_3d = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*
                  (*get_quad_z())(kk)*
                  (pow(u(ii,jj,kk),2)+pow(v(ii,jj,kk),2)+
                   pow(w(ii,jj,kk),2)))); 
         ke_2d = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_z())(jj)*
                  (pow(mean(u(ii,kk,jj),kk),2)+
                   pow(mean(w(ii,kk,jj),kk),2))))*length_y();
         ke_p = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*
                  (*get_quad_z())(kk)*
                  (pow(u(ii,jj,kk)-U_SCALE*pow(cosh(zz(kk)/JET_WIDTH),-2),2)+
                   pow(v(ii,jj,kk),2)+
                   pow(w(ii,jj,kk),2)))); 
         max_v = psmax(max(abs(v)));

         times_log[itercount-last_writeout-1] = time;
         ke_3d_log[itercount-last_writeout-1] = ke_3d;
         ke_2d_log[itercount-last_writeout-1] = ke_2d;
         ke_p_log[itercount-last_writeout-1] = ke_p;
         max_v_log[itercount-last_writeout-1] = max_v;
         max_rho_log[itercount-last_writeout-1] = mddiff;


         if ((time - nextplot) > -1e-5*plot_interval) {
            plot_now = true;
            plotnum = plotnum + 1;
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(w,"w_output",plotnum);
            write_array(*tracer[0],"rho_output",plotnum);
            lastplot = itercount;
            if (master()) {
               FILE * plottimes = fopen("plot_times.txt","a");
               assert(plottimes);
               fprintf(plottimes,"%.10g\n",time);
               fclose(plottimes);
               fprintf(stdout,"*");
            }
            nextplot = nextplot + plot_interval;
         } else {
            plot_now = false;
         }
         if (((itercount - lastplot) % 100 == 0) || plot_now) {
            double mu = psmax(max(abs(u(ii,jj,kk)-U_SCALE*pow(cosh(zz(kk)/JET_WIDTH),-2)))),
                   mw = psmax(max(abs(w)));
            if (master())
               fprintf(stdout,"%f [%d]: (%.2g, %.2g, %.2g: %.3g [%.3g] [%.3g]) %.2g\n",
                     time,itercount,mu,max_v,mw,ke_3d,ke_3d-ke_2d,ke_p,
                     mddiff);
            if (master()) {
               FILE * run_log = fopen("diagnostics.txt","a");
               assert(run_log);
               for (int i = 0; i < (itercount-last_writeout); i++) {
                  fprintf(run_log,
                        "%.10g %.10e %.10e %.10e %.10e %.10e %.10e\n",
                        times_log[i],ke_3d_log[i],ke_2d_log[i],
                        ke_3d_log[i]-ke_2d_log[i],ke_p_log[i],
                        max_v_log[i],
                        max_rho_log[i]);
               }
               fclose(run_log);
            }
            last_writeout = itercount;
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         /* Initialize u, w from read-in slice */
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* u */
            snprintf(filename,100,"u_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(u,filename,Nx,Ny,Nz);

            /* v */
            snprintf(filename,100,"v_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading v from %s\n",filename);
            read_array(v,filename,Nx,Ny,Nz);

            /* w */
            snprintf(filename,100,"w_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading w from %s\n",filename);
            read_array(w,filename,Nx,Ny,Nz);
            return;
         }
         /* Initialize a jet pointing in the x-direction */
         v = 0;
         w = 0;

         u = U_SCALE*pow(cosh(zz(kk)/JET_WIDTH),-2)+0*kk;

         /* And add a random perturbation */
         Normal<double> rnd(0,1); // Normal random number generator
         int myrank; // get MPI rank as seed, so processors don't repeat random numbers
         MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
         rnd.seed(myrank);
         /* Add random noise to the jet at a 1e-3 level */
         for (int i = u.lbound(firstDim); i <= u.ubound(firstDim); i++) {
            for (int j = u.lbound(secondDim); j <= u.ubound(secondDim); j++) {
               for (int k = u.lbound(thirdDim); k <= u.ubound(thirdDim); k++) {
                  double scale = u(i,j,k)*1e-3;
                  v(i,j,k) = DELTA*scale*rnd.random();
                  w(i,j,k) = scale*rnd.random();
                  u(i,j,k) += scale*rnd.random();
               }
            }
         }
         if (size_y() == 1) {
            /* If this is a 2D run, remove v */
            v = 0;
         }
         write_array(u,"u_output",0);
         write_array(v,"v_output",0);
         write_array(w,"w_output",0);
         write_reader(u,"u_output",true);
         write_reader(v,"v_output",true);
         write_reader(w,"w_output",true);
        
      }
      void init_tracer(int t_num, DTArray & rho) {
         /* Initialize the density and take the opportunity to write out the grid */
         assert(t_num == 0);
         /* Generate the phase differences in y.  Seeding is done per-processor,
            so this can change if we restart with a different number of
            processors */
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* rho */
            snprintf(filename,100,"rho_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(rho,filename,Nx,Ny,Nz);
            return;
         }
         rho = 0; // Zero perturbation density to start
         write_array(rho,"rho_output",0);
         write_reader(rho,"rho_output",true);
         
         /* Write out the grid for plotting in MATLAB */
         Array<double,3> grid(alloc_lbound(Nx,Ny,Nz), alloc_extent(Nx,Ny,Nz), alloc_storage(Nx,Ny,Nz));
         grid = xx(ii) + 0*jj + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = 0*ii + yy(jj) + 0*kk;
         write_array(grid,"ygrid"); write_reader(grid,"ygrid",false);
         grid = 0*ii + 0*jj + zz(kk);
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);
      }

      void forcing(double t, const DTArray & u, DTArray & u_f,
          const DTArray & v, DTArray & v_f, const DTArray & w, DTArray & w_f,
          vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
         /* There are three forcings to velocity:
            1) Traditional rotation
            2) Nontraditional rotation
            3) Density */

         /* As a reminder, y is the -vertical- coordinate and v is its velocity */
         u_f = ROT_F*w - ROT_B*v;
         w_f = -ROT_F*u;
         v_f = ROT_B*u - *(tracers[0]); // subtract perturbation density
         if (size_y() == 1) // Clear out vertical forcing if this is a 2D problem
            v_f = 0;

         *(tracers_f[0]) = BUOY_SCALE*v;

      }

      userControl() :
         plotnum(restart_sequence), plot_interval(.1/ROT_F),
         nextplot(plot_interval*(restart_sequence+1)), itercount(0), 
         lastplot(0),
         last_writeout(0),
         base_rho(Nz),phase_y(Ny),
         zz(Nz), yy(Ny), xx(split_range(Nx)) {
            compute_quadweights(size_x(),size_y(),size_z(),
                  length_x(),length_y(),length_z(),
                  type_x(),type_y(),type_z());
            xx = MIN_X + Lx*(ii+0.5)/Nx;
            yy = MIN_Y + Ly*(ii+0.5)/Ny;
            zz = MIN_Z + Lz*(ii+0.5)/Nz;
         }
};

int main(int argc, char ** argv) {
   /* For now, filtering parameters are controlled via this set
      of global variables.  It's ad-hoc, but it works.  The
      negative strength turns on the implicit hyperviscosity --
      otherwise it would use a sharp (exponential) cutoff */
   f_strength = -.25;
   f_order = 4;
   f_cutoff = 0.8;
   MPI_Init(&argc,&argv);
   if (master()) {
      fprintf(stdout,"Starting barotropic instability run.  Parameters:\n");
      fprintf(stdout,"Lx = %g, Lz = %g, Ly = %g\n",Lx,Lz,Ly);
      fprintf(stdout,"Nx = %d, Nz = %d, Ny = %d\n",Nx,Nz,Ny);
      fprintf(stdout,"U = %g, N0 = %g\nf = %g, b = %g\n",U_SCALE,BUOY_SCALE,ROT_F,ROT_B);
      fprintf(stdout,"Ro = %g, \\delta = %g, 1/FR = %f\n",ROSSBY,DELTA,FROUDE_INV);
      fprintf(stdout,"Jet width = %g\n",JET_WIDTH);
   }
   if (argc > 3) {
      /* Possibly restarting */
      if (strcmp("--restart",argv[1])==0) {
         if (master()) fprintf(stdout,"Restarting...\n");
         restarting = true;
         restart_time = atof(argv[2]);
         if (master()) fprintf(stdout,"Restart time %f\n",restart_time);
         restart_sequence = atoi(argv[3]);
         if (master()) fprintf(stdout,"Restart sequence %d\n",restart_sequence);
      }
   }
   userControl mycode;
   FluidEvolve<userControl> barotropic(&mycode);
   barotropic.initialize();
   barotropic.do_run(1e2/ROT_F/ROSSBY); /* Run for 1000 periods of 1/f */
   MPI_Finalize();
}
