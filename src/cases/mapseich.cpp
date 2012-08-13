/* Shoaling wave, with a mapped grid and all four walls no-slip */
#include "../Par_util.hpp"
#include <mpi.h>
#include "../BaseCase.hpp"
#include "../TArray.hpp"
#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include <math.h>
#include <stdio.h>
#include <blitz/array.h>
#include <random/normal.h> // Random number generation
#include <vector>

using namespace std;
using namespace TArrayn;
using namespace NSIntegrator;
using namespace ranlib;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;
const double g = 9.81; // m/s^2, gravity
const double rho_0 = 1; //  kg / L, fresh water at 20 C
const double pic_depth = -.025; // picnocline depth
const double pic_strength = .015; // strength
const double pic_width = .015;

// Parameters for the initial density disturbance
const double cos_height = .02;
//const double sech_width = 1;
//const double sech_loc = -3;

const double Lx = 0.375; // domain half-length
const double Lz = .075; // domain half-height
const double hill_size = .03; //0.025; // Hill size
const double hill_loc = -.05; // hill location, physical 
const double hill_width = .025;
const double right_slope = 0; // Slope of right boundary (inverse)
   // Salt diffusivity is O(10^-9) ~= 0 

class userControl : public BaseCase {
   public:
      int szx, szz, plotnum, itercount;
      double plot_interval, nextplot;


      DTArray * xgrid, * zgrid, * eta;

      int size_x() const { return szx; }
      int size_y() const { return 1; }
      int size_z() const { return szz; }

      DIMTYPE type_x() const { return FREE_SLIP; }
      DIMTYPE type_z() const { return NO_SLIP; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      double get_diffusivity(int t) const {
         return 1e-7;
      }
      double get_visco() const {
         return 1e-7;
      }

      int numActive() const { return 1; }

      double length_x() const { return 2*Lx; }
      double length_y() const { return 1; }
      double length_z() const { return 2*Lz; }

      bool is_mapped() const { return true; }
      void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
         // simple mapping, slope the bottom boundary
         xgrid = alloc_array(szx,1,szz);
         zgrid = alloc_array(szx,1,szz);
         Array<double,1> xx(split_range(szx)), zz(szz);
         // Define numerical Cheby coordinates.  Use the arcsin mapping to
         // stretch the grid towards uniformity
         //xx = cos(ii*M_PI/(szx-1));
         xx = 2*(ii+0.5)/szx - 1;
         zz = cos(ii*M_PI/(szz-1));

         xg = Lx*xx(ii) + 0*jj+0*kk;
         *xgrid = xg;

         yg = 0;

         zg = -Lz+Lz*zz(kk) + hill_size/2*(1-zz(kk))*(pow(cosh((xg(ii,jj,kk)-hill_loc)/hill_width),-2));
         //zg = Lz*zz(kk) + hill_size/4*(1-zz(kk))*(1+tanh((xx(ii)-hill_loc)/hill_width));
         *zgrid = zg;
         write_array(xg,"xgrid");
         write_reader(xg,"xgrid",false);
         write_array(zg,"zgrid");
         write_reader(zg,"zgrid",false);
      }


      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            /* Something's gone wrong, so abort */
            fprintf(stderr,"Tiny timestep returned, aborting\n");
            return -1;
         } else if (itercount < 10 && intime > 1e-1) {
            /* Cap the maximum timestep size */
            intime = 1e-1; // And process to make sure we land on a plot time
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
            vector<DTArray *> tracer, DTArray & pressure) {
         /* Write out velocities and density if we've passed the designated
            plot time */
         bool plotted = false;
         if (time > (nextplot - 1e-8*fabs(nextplot))) {
            nextplot = nextplot + plot_interval;
            if (master()) fprintf(stderr,"*");
            plotnum++;
            write_array(u,"u_output",plotnum);
            write_array(w,"w_output",plotnum);
            write_array(*(tracer[0]),"rho_output",plotnum);
            plotted = true;
         }
         itercount++;
         if (!(itercount % 1) || plotted) {
            /* Print out some diagnostic information */
            double maxrho = pvmax(*tracer[0]);
            double minrho = pvmin(*tracer[0]);
            double mu = psmax(max(abs(u))),
                   mw = psmax(max(abs(w))),
                   rhodiff = maxrho - minrho;
            if (master()) {
               fprintf(stderr,"%f (%d): (%.2g, %.2g), (%.2g)\n",
                  time, itercount, mu, mw, rhodiff-pic_strength);
            }
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         //do_mapping(u,v,w);
         v = 0;
         u = 0;
         w = 0;
         write_reader(u,"u_output",true);
         write_reader(w,"w_output",true);
         write_array(u,"u_output",0);
         write_array(w,"w_output",0);
      }
      void init_tracer(int t_num, DTArray & rho) {
         
         // The base stratification is pic_strength/2*tanh((zz(kk)-pic_depth)/pic_width)
         // but the grid is mapped between the i/k indices.

         rho = -pic_strength/2*tanh((1/pic_width) *
                  ((*zgrid)(ii,jj,kk) - pic_depth - cos_height*sin((*xgrid)(ii,jj,kk)*M_PI/Lx/2)));
                     
         write_array(rho,"rho_output",0);
         write_reader(rho,"rho_output",true);
      }
      
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         /* Simple gravity-based body forcing, with gravity pointing
            downwards in the y-direction. */
         u_f = 0; v_f = 0;
         w_f = -g*(rho_0 + *tracers[0]);
      }
      void tracer_forcing(double t, DTArray & u, DTArray & v,
            DTArray & w, vector<DTArray *> & tracers_f) {
         /* Forcing on the tracers themselves.  Since rho is a passive density,
            there is none. */
         *tracers_f[0] = 0;
      }

      userControl() : 
         szx(2048), szz(512),
         //szx(512), szz(128),
//         szx(48), szz(16),
//         szx(24), szz(8),
         plotnum(0), plot_interval(1), nextplot(plot_interval),
         itercount(0) {
         }
};

int main() {
   MPI_Init(0,0);
   userControl mycode;
   FluidEvolve<userControl> slosh(&mycode);
   slosh.initialize();
   slosh.do_run(100);
   MPI_Finalize();
   return 0;
}
