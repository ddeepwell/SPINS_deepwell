/* Wave test case -- long periodic tank with a solitary wave */
#include "../BaseCase.hpp"
#include "../TArray.hpp"
#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include <mpi.h>
#include <stdio.h>
#include <blitz/array.h>
#include <vector>

using namespace std;
using namespace NSIntegrator;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

class userControl : public BaseCase {
   public:
      int szx, szy, plotnum, itercount;
      double Lx, Ly, plot_interval, nextplot;

      /* Variables to control stratification */
      double pic_width, height;
      double width, amplitude;
      /* (reduced) gravity */
      double g;

      int size_x() const { return szx; }
      int size_y() const { return szy; }
      int size_z() const { return 1; }

      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      /* One active tracer -- density */
      int numActive() const { return 1; }

      double length_x() const { return Lx; }
      double length_y() const { return Ly; }
      double length_z() const { return 1; }

      /* This (and writeout for analysis) sould probably be abstracted into
         another mixin class */
      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            /* Something's gone wrong, so abort */
            fprintf(stderr,"Tiny timestep returned, aborting\n");
            return -1;
         } else if (intime > .1) {
            /* Cap the maximum timestep size */
            intime = .1; // And process to make sure we land on a plot time
         }
         /* Calculate how many timesteps we'll take until we pass the
            next plottime. */
         double until_plot = nextplot - now;
         double steps = ceil(until_plot / intime); 
         double real_until_plot = steps*intime;

         if (fabs(until_plot - real_until_plot) < 1e-7) {
            /* We'll hit close enough to the plot point, so good enough */
            return intime;
         } else {
            /* Adjust */
            return (until_plot / steps);
         }
      }

      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer) {
         /* Write out velocities and density if we've passed the designated
            plot time */
         bool plotted = false;
         if (time > (nextplot - 1e-8*fabs(nextplot))) {
            nextplot = nextplot + plot_interval;
            fprintf(stderr,"*");
            plotnum++;
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(*(tracer[0]),"rho_output",plotnum);
            plotted = true;
         }
         itercount++;
         if (!(itercount % 25) || plotted) {
            /* Print out some diagnostic information */
            fprintf(stderr,"%f: (%.2g, %.2g: %.2f), (%.2g)\n",
                  time, max(abs(u)), max(abs(v)), max(abs(*tracer[0])),
                  mean(u*u+v*v));
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         u = 0; v=0; w=0;
         write_reader(u,"u_output",true);
         write_reader(v,"v_output",true);
      }

      void init_tracer(int t_num, DTArray & rho) {
         /* Initialize density to a "tilted" interface, as if an at-rest-but-
            tilted tank were suddenly returned to the horizontal */
         assert(t_num == 0);
         Array<double,1> xx(szx), yy(szy);
         xx = -Lx/2 + Lx*(ii+0.5)/szx;
         yy = -Ly/2 + Ly*(ii+0.5)/szy;
         Array<double,3> grid(szx, szy, 1);
         grid = xx(ii) + 0*jj + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = 0*ii + yy(jj) + 0*kk;
         write_array(grid,"ygrid"); write_reader(grid,"ygrid",false);

         rho = 0.5 - 
            0.5*tanh(1/pic_width*(yy(jj) - height -
                     amplitude/pow(cosh(xx(ii)/width),2))) + 0*kk;
         write_array(rho,"rho_output",0); write_reader(rho,"rho_output",true);
      }

      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         /* Simple gravity-based body forcing, with gravity pointing
            downwards in the y-direction. */
         u_f = 0; w_f = 0;
         v_f = -g*(*tracers[0]);
      }
      void tracer_forcing(double t, const DTArray & u, const DTArray & v,
            const DTArray & w, vector<DTArray *> & tracers_f) {
         /* Forcing on the tracers themselves.  Since rho is a passive density,
            there is none. */
         *tracers_f[0] = 0;
      }

      userControl() : 
         szx(1024), szy(64), Lx(80), Ly(1),
         plotnum(0), plot_interval(0.1), nextplot(plot_interval),
         itercount(0), g(.2), 
         height(-0.25), pic_width(0.05),
         width(2), amplitude(0.2) {
         }
};

int main() {
   MPI_Init(0,0);
   userControl mycode;
   FluidEvolve<userControl> slosh(&mycode);
   slosh.initialize();
   slosh.do_run(150);
   MPI_Finalize();
   return 0;
}
