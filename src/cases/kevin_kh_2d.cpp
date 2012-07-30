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
#define Lx 2000
#define Nx 12500
#define MIN_X -500

#define Lz 100
#define Nz 3000
#define MIN_Z -100

#define Ly 1
#define Ny 1
#define MIN_Y (-Ly/2.0)

#define G 9.81

#define bg_u 0.63457396

using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

class userControl : public BaseCase {
   public:
      int plotnum, itercount, lastplot;
      bool plot_now;
      double plot_interval, nextplot;
      double start_rho_spread;

      Array<double,1> xx,yy,zz;

      int size_x() const { return Nx; }
      int size_y() const { return Ny; }
      int size_z() const { return Nz; }

      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_y() const { return PERIODIC; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      double length_x() const { return Lx; }
      double length_y() const { return Ly; }
      double length_z() const { return Lz; }

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
         } else if (intime > .5) {
            /* Cap the maximum timestep size */
            intime = .5;
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
         read_2d_slice(rho,"input_d",Nx,Nz);
         start_rho_spread = pvmax(rho) - pvmin(rho);
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

      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         /* Velocity forcing */
         double freq = 0.1;
         u_f = 0; v_f = 0;
         w_f = -G*(*tracers[0]) +
               0.0001*sin(freq*t)*pow(cosh((xx(ii)+450)/10),-2) *
                  (pow(cosh((zz(kk)+9)/4),-2) + pow(cosh((zz(kk)+15)/4),-2));
      }
      void tracer_forcing(double t, const DTArray & u, const DTArray & v,
            const DTArray & w, vector<DTArray *> & tracers_f) {
         *tracers_f[0] = 0;
      }

      userControl() :
         plotnum(0), plot_interval(6),
         nextplot(plot_interval), itercount(0), lastplot(0),
         zz(Nz), yy(Ny), xx(split_range(Nx)) {
            xx = MIN_X + Lx*(ii+0.5)/Nx;
            yy = MIN_Y + Ly*(ii+0.5)/Ny;
            zz = MIN_Z + Lz*(ii+0.5)/Nz;
         }
};

int main() {
   MPI_Init(0,0);
   userControl mycode;
   EasyFlow kevin_kh(&mycode);
   kevin_kh.initialize();
   kevin_kh.do_run(3000);
}
