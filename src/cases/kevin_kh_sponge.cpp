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

#define Ly 1
#define Ny 1
#define MIN_Y (-Ly/2.0)

#define G 9.81

#define bg_u 0.63457396

#define sponge_start 450
#define sponge_end 500
#define sponge_length 10
#define sponge_strength 5

using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

int myrank = -1;

bool restarting = false;
int restart_sequence = 0;
double restart_time = 0;

double times_record[100];

class userControl : public BaseCase {
   public:
      int plotnum, itercount, lastplot;
      bool plot_now;
      double plot_interval, nextplot;
      double start_rho_spread;

      Array<double,1> xx,yy,zz;
      Array<double,1> base_rho;

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

      /* Data analysis and output */
      void tracer_analysis(double time, int t_num, DTArray & dens) {
         assert(t_num == 0);
         /* Write out density */
         if (plot_now) 
            write_array(dens,"rho_output",plotnum);
         if ((itercount - lastplot) % 10 == 0 || plot_now) {
            double mddiff = pvmax(dens) - pvmin(dens);
            if (master())
               fprintf(stdout,"(%.2g)\n",(mddiff-start_rho_spread)/start_rho_spread);
         }
      }
      void vel_analysis(double time, DTArray & u, DTArray & v, DTArray & w) {
         itercount = itercount + 1;
         if ((time - nextplot) > -1e-5*plot_interval) {
            plot_now = true;
            plotnum = plotnum + 1;
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(w,"w_output",plotnum);
            lastplot = itercount;
            if (master())
               fprintf(stdout,"*");
            nextplot = nextplot + plot_interval;
         } else {
            plot_now = false;
         }
         if (((itercount - lastplot) % 10 == 0) || plot_now) {
            double mu = psmax(max(abs(u-bg_u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w))),
                   ke = pssum(sum((u-bg_u)*(u-bg_u) + v*v + w*w)*Lx*Ly*Lz/(Nx*Ny*Nz));
            if (master())
               fprintf(stdout,"%f [%d]: (%.2g, %.2g, %.2g: %.3g) ",
                     time,itercount,mu,mv,mw,ke);
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
         /* Initialize the density and take the opportunity to write out the grid */
         assert(t_num == 0);
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
         read_2d_slice(rho,"input_d",Nx,Nz);
         start_rho_spread = pvmax(rho) - pvmin(rho);
         /* Cheat and read the first rho as the baseline, to force to in the
            sponge layer */
         Array<double,3> base_rho_3d(1,1,Nz);
         read_2d_slice(base_rho_3d,"input_d",Nx,Nz);
         base_rho = base_rho_3d(0,0,blitz::Range::all());
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
         /* Add a second, subharmonic component to the forcing to possibly
            trip KH-pairing */
         w_f = -G*(*tracers[0]) +
               0.0001*sin(freq*t)*pow(cosh((xx(ii)+300)/10),-2) *
                  (pow(cosh((zz(kk)+9)/4),-2) + pow(cosh((zz(kk)+15)/4),-2)) +
               0.00005*cos(freq*t/2)*pow(cosh((xx(ii)+300)/10),-2) *
                  (pow(cosh((zz(kk)+9)/4),-2) + pow(cosh((zz(kk)+15)/4),-2));

         /* Apply sponge layer */
         u_f = -sponger(ii)*(u(ii,jj,kk)-bg_u);
         v_f = -sponger(ii)*v(ii,jj,kk);
         w_f = w_f - sponger(ii)*w(ii,jj,kk);
         *(tracers_f[0]) = -sponger(ii)*((*tracers[0])(ii,jj,kk) - base_rho(kk));
      }

      userControl() :
         plotnum(0), plot_interval(6),
         nextplot(plot_interval), itercount(0), lastplot(0),
         base_rho(Nz),
         zz(Nz), yy(Ny), xx(split_range(Nx)) {
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
}
