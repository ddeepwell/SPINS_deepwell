/* Growth of instabilities in ISWs with shear-unstable regions, for the
   SHARCNET Small Dedicated Resources (SDR) grant on saw.sharcnet.

   This code uses input data supplied from Lamb's IGW model, interpolated
   to the spectral grid with MATLAB code. */


#include "../Science.hpp"
#include "../TArray.hpp"
#include "../Par_util.hpp"
#include "../NSIntegrator.hpp"
#include "../BaseCase.hpp"
#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <random/uniform.h>
#include <random/normal.h>
#include <time.h>


/* This is the DIMENSIONLESS version of internal_wave_shear,
   which has a fluid depth of 1 unit, g as 1 unit, and a
   normalized top-to-bottom density difference of 1. */

clock_t cputime_start = 0;
clock_t cputime_last = 0;
double Lx;
int Nx;
#define MIN_X (-Lx/2)

double Lz;
int Nz;
#define MIN_Z (-Lz)

/* We want to use the same code to run 2D and 3D cases.
   Physically, we want a spanwise length of one-to-two
   KH-billow wavelengths [O(30m)], but the exact length
   and number of points will depend on the case.  Leave
   the spanwise parameters as variables, to be read in
   from a configuration file */

int Ny;
double Ly;

#define G 1

/* The background (reference-frame) velocity can be
   derived from the initial wave field */
double bg_u;

/* Amplitude, location, and frequency of forcing are
   problem-dependant */
double AMPLITUDE, FORCING_LOC, FORCING_FREQ;

/* Input viscosity */
double in_viscosity = 0;

/* We want a sponge layer at the end of the domain to damp
   out billows as they leave, so they do not re-enter.

   A possible future extension of this would be to have
   perturbation forcing for only a fixed time and allow
   the billows to re-enter, to simulate what would happen
   to a wave-train of such waves */

#define sponge_start (Lx/2*0.8)
#define sponge_end (Lx/2)
// Parentheses you total uber idiot!
#define sponge_length (Lx/100)
#define sponge_strength 100
//#define sponge_strength 40

/* Finally, we can output a lot more often in 2D than in 3D */
double input_plot_interval;

using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

using ranlib::Normal;

int myrank = -1;

bool restarting = false;
int restart_sequence = 0;
double restart_time = 0;

/* Log arrays, to store some diagnostic information on a per-timestep
   resolution */
double times_log[100], ke_2d_log[100], ke_3d_log[100], max_v_log[100],
       max_drho_log[100], pe_log[100];

/* Filenames for input u/w/rho reading */
char RHO_FILENAME[1000], U_FILENAME[1000], W_FILENAME[1000];

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

      // We're not resolving viscous scales, so turn off
      // diffusivity and viscosity
      double get_viscosity() const {
         return in_viscosity; // Nearly nonviscious, save for filtering
      }
      double get_diffusivity(int t) const {
         return in_viscosity; // Nondiffusive tracer, save for filtering
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
         double mddiff, ke_2d, ke_3d, pe, max_v;
         clock_t cputime = clock();

         /* For output, write 3D kinetic energy, 2D kinetic energy,
            maximum spanwise velocity, potential energy less
            the background state, and total density difference
            which is a diagnostic of the filtering (since filtering
            does not preserve maximums) */
         mddiff = pvmax(*tracer[0]) - pvmin(*tracer[0]);
         ke_3d = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*
                  (*get_quad_z())(kk)*
                  (pow(u(ii,jj,kk)-bg_u,2)+pow(v(ii,jj,kk),2)+
                   pow(w(ii,jj,kk),2))))/2; 
         ke_2d = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_z())(jj)*
                  (pow(mean(u(ii,kk,jj),kk)-bg_u,2)+
                   pow(mean(w(ii,kk,jj),kk),2))))*length_y()/2;
         pe    = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*
                  (*get_quad_z())(kk)*((*tracer[0])(ii,jj,kk)-
                     base_rho(kk))*G*zz(kk)));
         max_v = psmax(max(abs(v)));

         times_log[itercount-last_writeout-1] = time;
         ke_3d_log[itercount-last_writeout-1] = ke_3d;
         ke_2d_log[itercount-last_writeout-1] = ke_2d;
         max_v_log[itercount-last_writeout-1] = max_v;
         max_drho_log[itercount-last_writeout-1] = mddiff;
         pe_log   [itercount-last_writeout-1] = pe;


         if ((time - nextplot) > -1e-5*plot_interval) {
            plot_now = true;
            plotnum = plotnum + 1;
            write_array(u,"u",plotnum);
            if (Ny > 1)
               write_array(v,"v",plotnum);
            write_array(w,"w",plotnum);
            write_array(*tracer[0],"rho",plotnum);
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
               fprintf(stdout,"%f [%d]: (%.2g, %.2g, %.2g: %.3g [%.3g] + %.3g) %.2g\n",
                     time,itercount,mu,max_v,mw,ke_3d,ke_3d-ke_2d,pe,
                     (mddiff-start_rho_spread)/start_rho_spread);
            if (master()) {
               if (cputime_start == 0) {
                  cputime_start == cputime;
               }
               // memory debugging -- write out
               // the contents of /proc/self/statm to a log file
               // to diagnose memory issues
               FILE * mem_log = fopen("mem_log.txt","a");
               assert(mem_log);
               int mem_size, mem_res, mem_share;
               FILE * statm = fopen("/proc/self/statm","r");
               assert(statm);
               fscanf(statm,"%d %d %d",&mem_size,&mem_res,&mem_share);
               fclose(statm);
               fprintf(mem_log,"%d: sz: %d res: %d share: %d time: %g incr: %g\n",itercount,
                     mem_size,mem_res,mem_share,
                     float(cputime-cputime_start)/CLOCKS_PER_SEC/itercount,
                     float(cputime-cputime_last)/CLOCKS_PER_SEC);
               fclose(mem_log);
               cputime_last = cputime;
               FILE * run_log = fopen("diagnostics.txt","a");
               assert(run_log);
               for (int i = 0; i < (itercount-last_writeout); i++) {
                  fprintf(run_log,
                        "%.10g %.10e %.10e %.10e %.10e %.10e %.10e\n",
                        times_log[i],ke_3d_log[i],ke_2d_log[i],
                        ke_3d_log[i]-ke_2d_log[i],pe_log[i],
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
         /* We must first read-in the initial u state, since we have
            to subtract the reference-frame speed when making kinetic
            energy calculations.  The reference-frame speed is given
            by the mean initial u, or alternately u(1,1,1). */
         read_2d_slice(u,U_FILENAME,Nx,Nz);
         if (master()) fprintf(stderr,"1\n");
         if (master()) {
            bg_u = u(0,0,0);
            MPI_Bcast(&bg_u,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
         } else {
            MPI_Bcast(&bg_u,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
         }
         if (master()) fprintf(stderr,"2\n");
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* u */
            snprintf(filename,100,"u.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(u,filename,Nx,Ny,Nz);

            /* v */
            if (Ny > 1) {
               snprintf(filename,100,"v.%d",restart_sequence);
               if (master()) fprintf(stdout,"Reading v from %s\n",filename);
               read_array(v,filename,Nx,Ny,Nz);
            } else {
               v = 0;
            }
               

            /* w */
            snprintf(filename,100,"w.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading w from %s\n",filename);
            read_array(w,filename,Nx,Ny,Nz);
            return;
         } 
         read_2d_slice(w,W_FILENAME,Nx,Nz);
         if (master()) fprintf(stderr,"3\n");
         /* And v is 0 */
         v = 0; 
         /* Write out initial values */
         write_array(u,"u",0);
         write_reader(u,"u",true);
         if (Ny > 1) {
            write_array(v,"v",0);
            write_reader(v,"v",true);
         }
         write_array(w,"w",0);
         write_reader(w,"w",true);
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
            Normal<double> rnd(0,1);
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
            snprintf(filename,100,"rho.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(rho,filename,Nx,Ny,Nz);
            Array<double,3> base_rho_3d(1,1,Nz);
            read_2d_slice(base_rho_3d,RHO_FILENAME,Nx,Nz);
            base_rho = base_rho_3d(0,0,blitz::Range::all());
            start_rho_spread = pvmax(base_rho) - pvmin(base_rho);
            return;
         }
         read_2d_slice(rho,RHO_FILENAME,Nx,Nz);
         /* Cheat and read the first rho as the baseline, to force to in the
            sponge layer */
         Array<double,3> base_rho_3d(1,1,Nz);
         read_2d_slice(base_rho_3d,RHO_FILENAME,Nx,Nz);
         base_rho = base_rho_3d(0,0,blitz::Range::all());
         start_rho_spread = pvmax(base_rho) - pvmin(base_rho);
/*         if( master() ){
            cout << base_rho_3d;
            cout << base_rho;
         }
         MPI_Finalize(); exit(1);*/
         write_array(rho,"rho",0);
         write_reader(rho,"rho",true);
         
         Array<double,3> grid(alloc_lbound(Nx,Ny,Nz), alloc_extent(Nx,Ny,Nz), alloc_storage(Nx,Ny,Nz));
         grid = xx(ii) + 0*jj + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         if (Ny > 1) {
            grid = 0*ii + yy(jj) + 0*kk;
            write_array(grid,"ygrid"); write_reader(grid,"ygrid",false);
         }
         grid = 0*ii + 0*jj + zz(kk);
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);
      }

      void forcing(double t, const DTArray & u, DTArray & u_f,
          const DTArray & v, DTArray & v_f, const DTArray & w, DTArray & w_f,
          vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
         /* Build sponge layer */
         Array<double,1> sponger(split_range(Nx));
         sponger = sponge_strength/2*
               (tanh((xx-sponge_start)/sponge_length) -
                tanh((xx-sponge_end)/sponge_length));
//         sponger = sponge_strength/2*
//            (tanh((xx(ii)-sponge_start)/sponge_length) -
 //            tanh((xx(ii)-sponge_end)/sponge_length));
         /* Velocity forcing */
         u_f = 0; v_f = 0;
         /* Add a random component to w-forcing with specified temporal
            frequency, to provoke KH-billows in the unstable wave.

            The forcing is centered at z=FORCING_LOC (varies per wave)
            and has frequency omega=FORCING_FREQ (radians/sec).  The
            amplitude is (obviously) AMPLITUDE, and in addition there is
            a 10% white-noise applied to tha amplitude in the spanwise to
            stimulate any three-dimensionalization we might see.  Not too
            much luck with that so far.

            The forcing is compact in x (20-wide) and z (8-wide), and it
            is centered at z=-300, which should be just outside of the
            wave proper. */
         w_f = -G*((*tracers[0])(ii,jj,kk)-base_rho(kk)) +
               AMPLITUDE*(1+0.1*phase_y(jj))*sin(FORCING_FREQ*t)*
                  pow(cosh((xx(ii)+7.0*Lx/20)/(Lx/100)),-2) *
                  pow(cosh((zz(kk)-FORCING_LOC)/(Lz/25)),-2);

         /* Apply sponge layer */
         u_f = -sponger(ii)*(u(ii,jj,kk)-bg_u);
         v_f = -sponger(ii)*v(ii,jj,kk);
         w_f = w_f - sponger(ii)*w(ii,jj,kk);
         *(tracers_f[0]) = -sponger(ii)*((*tracers[0])(ii,jj,kk) - base_rho(kk));
      }

      userControl() :
         plotnum(restart_sequence), plot_interval(input_plot_interval),
         nextplot(plot_interval*(plotnum+1)), itercount(0), lastplot(0),
         last_writeout(0),
         base_rho(Nz),phase_y(Ny),
         zz(Nz), yy(Ny), xx(split_range(Nx)) {
            compute_quadweights(size_x(),size_y(),size_z(),
                  length_x(),length_y(),length_z(),
                  type_x(),type_y(),type_z());

               for (int i = xx.lbound(firstDim);
                     i <= xx.ubound(firstDim);
                     i++) {
                  xx(i) = i;
               }
               xx = xx + 0.5;
               xx = xx*Lx/Nx;
               xx = xx + MIN_X;
//            xx = MIN_X + Lx*(double(ii)+0.5)/Nx;
            yy = (-Ly/2) + Ly*(ii+0.5)/Ny;
            zz = MIN_Z + Lz*(ii+0.5)/Nz;
         }
};

int main(int argc, char ** argv) {
   /* Set implicit filtering */
   f_strength = -.25;
   f_order = 4;
   f_cutoff = 0.8;
   MPI_Init(&argc,&argv);
   /* Since we have a few parameters to look at, open a configuration
      file */
   FILE * config_file = fopen("wave_config","r");
   assert(config_file);
   /* Configuration file specification:
      RHO_FILENAME
      U_FILENAME
      W_FILENAME
      Ny
      Ly
      FORCING_FREQUENCY
      FORCING_LOC
      AMPLTITUDE
   ---
   None of the filenames should include whitespace characters. */

   fscanf(config_file,"%s %s %s %d %lf %d %lf %d %lf %lf %lf %lf %lf %lf",
         RHO_FILENAME,U_FILENAME,W_FILENAME,
         &Nx, &Lx,
         &Ny, &Ly, 
         &Nz, &Lz,
         &FORCING_FREQ, &FORCING_LOC, &AMPLITUDE,
         &input_plot_interval, &in_viscosity);
   fclose(config_file);
   if (master()) {
      fprintf(stdout,"Received filenames:\nDensity: %s\nU: %s\nW: %s\n",
            RHO_FILENAME, U_FILENAME, W_FILENAME);
      fprintf(stdout,"Streamwise grid parameters: Length %f (%d points)\n",Lx,Nx);
      fprintf(stdout,"Spanwise grid parameters: Length %f (%d points)\n",Ly,Ny);
      fprintf(stdout,"Depth grid parameters: Depth %f (%d points)\n",Lz,Nz);
      fprintf(stdout,"Forcing parameters: Frequency %g, Amplitude %g, Location %f\n",
            FORCING_FREQ,AMPLITUDE,FORCING_LOC);
      fprintf(stdout,"Input viscosity: %g", in_viscosity);
      if (in_viscosity == 0)
         fprintf(stdout," (inviscid)\n");
      else
         fprintf(stdout,"\n");
      fprintf(stdout,"Plot time interval: %f\n",input_plot_interval);
   }
   assert(Ly > 0 && Ny > 0 && input_plot_interval > 0);
   assert(in_viscosity >= 0);
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
   cputime_last = clock();
   FluidEvolve<userControl> kevin_kh(&mycode);
   kevin_kh.initialize();
   kevin_kh.do_run(80);
   MPI_Finalize();
}
