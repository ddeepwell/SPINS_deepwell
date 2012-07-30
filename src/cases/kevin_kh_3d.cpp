/* Showcase of growth of convective instability in a large internal wave, using data
   read in from another source (IGW) */

#include "../Science.hpp"
#include "../TArray.hpp"
#include "../Par_util.hpp"
#include "../NSIntegrator.hpp"
#include "../BaseCase.hpp"
#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <random/uniform.h>

/* Domain parameters.  These MUST MATCH the data read in, currently given by matlab
   interpolation of igw data */

/* Additionally, it's pretty much impossible to resolve all scales at a super-
   fine level.  Filtering keeps the calculation stable, but aliasing error
   is obvious after the KH billows form and grow.  So, experiment with
   an eddy viscosity/diffusivity term to compensate */
#define Lx 1000
#define Nx 4096
#define MIN_X -500

#define Lz 100
#define Nz 768
#define MIN_Z -100

/* Ly=8, Ny=32 did not produce very significant 3D behaviour inside
   the wave.  So, to check if we're suppressing an instability, double
   the length and number of points spanwise */
#define Ly 16
#define Ny 1
#define MIN_Y (-Ly/2.0)

#define G 9.81

#define bg_u 0.63457396
#define AMPLITUDE 2e-3

#define sponge_start 450
#define sponge_end 500
#define sponge_length 10
#define sponge_strength 5

using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

using ranlib::Uniform;

int myrank = -1;

bool restarting = false;
int restart_sequence = 0;
double restart_time = 0;

double times_log[100], ke_2d_log[100], ke_3d_log[100], max_v_log[100],
       max_drho_log[100];

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

      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_y() const { return PERIODIC; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      double length_x() const { return Lx; }
      double length_y() const { return Ly; }
      double length_z() const { return Lz; }

      double get_visco() const {
         // Physical viscosity of 1e-6 was too low.  Try 10x that.
         // 1e-5 was still too low, giving balooning filtering to 10%
         // of density difference.  Try another factor of 10
         return 0; // Nearly nonviscious, save for filtering
//         return 1e-5; // Eddy viscosity
      }
      double get_diffusivity(int t) const {
         return 0; // Nondiffusive tracer, save for filtering
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
         } else if (intime > .1) {
            /* Cap the maximum timestep size */
            intime = .1;
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
         double mddiff, ke_2d, ke_3d, max_v;

         mddiff = pvmax(*tracer[0]) - pvmin(*tracer[0]);
         ke_3d = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*
                  (*get_quad_z())(kk)*
                  (pow(u(ii,jj,kk)-bg_u,2)+pow(v(ii,jj,kk),2)+
                   pow(w(ii,jj,kk),2)))); 
         ke_2d = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_z())(jj)*
                  (pow(mean(u(ii,kk,jj),kk)-bg_u,2)+
                   pow(mean(w(ii,kk,jj),kk),2))))*length_y();
         max_v = psmax(max(abs(v)));

         times_log[itercount-last_writeout-1] = time;
         ke_3d_log[itercount-last_writeout-1] = ke_3d;
         ke_2d_log[itercount-last_writeout-1] = ke_2d;
         max_v_log[itercount-last_writeout-1] = max_v;
         max_drho_log[itercount-last_writeout-1] = mddiff;


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
         if (((itercount - lastplot) % 1 == 0) || plot_now) {
            double mu = psmax(max(abs(u-bg_u))),
                   mw = psmax(max(abs(w)));
            if (master())
               fprintf(stdout,"%f [%d]: (%.2g, %.2g, %.2g: %.3g [%.3g]) %.2g\n",
                     time,itercount,mu,max_v,mw,ke_3d,ke_3d-ke_2d,
                     (mddiff-start_rho_spread)/start_rho_spread);
            if (master()) {
               FILE * run_log = fopen("diagnostics.txt","a");
               assert(run_log);
               for (int i = 0; i < (itercount-last_writeout); i++) {
                  fprintf(run_log,
                        "%.10g %.10e %.10e %.10e %.10e %.10e\n",
                        times_log[i],ke_3d_log[i],ke_2d_log[i],
                        ke_3d_log[i]-ke_2d_log[i],
                        max_v_log[i],
                        max_drho_log[i]);
               }
               fclose(run_log);
            }
            last_writeout = itercount;
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         /* Initialize u, w from read-in slice */
         if (master()) fprintf(stderr,"Initializing velocities\n");
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
         read_2d_slice(u,"input_u",Nx,Nz);
         read_2d_slice(w,"input_w",Nx,Nz);
         /* And v is 0 */
         v = 0; 
         /* Write out initial values */
         write_array(u,"u_output",0);
         write_reader(u,"u_output",true);
         write_array(v,"v_output",0);
         write_reader(v,"v_output",true);
         write_array(w,"w_output",0);
         write_reader(w,"w_output",true);
      }
      void init_tracer(int t_num, DTArray & rho) {
         if (master()) fprintf(stderr,"Initializing tracer\n");
         /* Initialize the density and take the opportunity to write out the grid */
         assert(t_num == 0);
         /* Generate the phase differences in y.  Seeding is done per-processor,
            so this can change if we restart with a different number of
            processors */
         /* ETA: Seeding is done -globally-, since the random noise here is
            strictly spanwise.  We want the same phase generated everywher */
         {
            Uniform<double> rnd;
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
            rnd.seed(0);
            for (int i = phase_y.lbound(firstDim); 
                  i <= phase_y.ubound(firstDim); i++) {
               phase_y(i) = rnd.random();
            }
            if (Ny == 1) phase_y(0) = 0;
         }
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* rho */
            snprintf(filename,100,"rho_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(rho,filename,Nx,Ny,Nz);
            Array<double,3> base_rho_3d(1,1,Nz);
            read_2d_slice(base_rho_3d,"input_d",Nx,Nz);
            base_rho = base_rho_3d(0,0,blitz::Range::all());
            start_rho_spread = pvmax(base_rho) - pvmin(base_rho);
            return;
         }
         read_2d_slice(rho,"input_d",Nx,Nz);
         /* Cheat and read the first rho as the baseline, to force to in the
            sponge layer */
         Array<double,3> base_rho_3d(1,1,Nz);
         read_2d_slice(base_rho_3d,"input_d",Nx,Nz);
         base_rho = base_rho_3d(0,0,blitz::Range::all());
         start_rho_spread = pvmax(base_rho) - pvmin(base_rho);
/*         if( master() ){
            cout << base_rho_3d;
            cout << base_rho;
         }
         MPI_Finalize(); exit(1);*/
         write_array(rho,"rho_output",0);
         write_reader(rho,"rho_output",true);
         
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
         /* Build sponge layer */
         Array<double,1> sponger(split_range(Nx));
         sponger = sponge_strength/2*
            (tanh((xx(ii)-sponge_start)/sponge_length) -
             tanh((xx(ii)-sponge_end)/sponge_length));
         /* Velocity forcing */
         double freq = 0.1;
         u_f = 0; v_f = 0;
         /* Location changed from -450 to -300 to show KH billows earlier */
         /* A sin(yy) term in the forcing would create a phase error, but
            this might not excite all harmonics.  Instead, generate a random
            phase that is time-constant (see init_tracer) and use it at each
            forcing step with a small multiplier (here 0.05), allowing a
            maximum spatial departure of 0.05/freq*u_bg ~= 0.3, on the order
            of one grid cell */ 
         /* ETA: 0.05 was too small to show much effect, so crank the forcing up
            to 0.5 radians - close to an O(1) perturbation in 3D. */
         /* To finish this once and for all, make it a 2_PI phase difference. */
         w_f = -G*(*tracers[0]) +
               AMPLITUDE*sin(freq*t+2*M_PI*phase_y(jj))*
                  pow(cosh((xx(ii)+300)/10),-2) *
                  (pow(cosh((zz(kk)+9)/4),-2) + pow(cosh((zz(kk)+15)/4),-2));

         /* Apply sponge layer */
         u_f = -sponger(ii)*(u(ii,jj,kk)-bg_u);
         v_f = -sponger(ii)*v(ii,jj,kk);
         w_f = w_f - sponger(ii)*w(ii,jj,kk);
         *(tracers_f[0]) = -sponger(ii)*((*tracers[0])(ii,jj,kk) - base_rho(kk));
      }

      userControl() :
         plotnum(restart_sequence), plot_interval(60),
         nextplot(plot_interval*(plotnum+1)), itercount(0), lastplot(0),
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
   /* Set implicit filtering */
   f_strength = -.25;
   f_order = 4;
   f_cutoff = 0.8;
   MPI_Init(&argc,&argv);
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
   FluidEvolve<userControl> kevin_kh(&mycode);
   kevin_kh.initialize();
   kevin_kh.do_run(3000);
   MPI_Finalize();
}
