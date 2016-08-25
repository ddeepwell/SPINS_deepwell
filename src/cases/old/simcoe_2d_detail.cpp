/* Shoaling wave, with a mapped grid and all four walls no-slip */
#include "../Par_util.hpp"
#include <mpi.h>
#include "../BaseCase.hpp"
#include "../TArray.hpp"
#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include "../Science.hpp"
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
const double rho_0 = 1000; //  kg / L, fresh water at 20 C

const double lin_t = 5; // Linear stratification: degC over depth
const double pic_t = 10 ; // Two-layer component, degC over thermocline
const double pic_H = -15;
const double pic_W = 2;
const double bkg_t = 5; // Background temperature; degC

// Parameters for the initial density disturbance
const double seiche_height = 5.0; // left-to-right disturbance, in m

// Grid parameters
const double Lx = 1500.0; // Domain length
const double xmin = -500; // Minimum x
const double Lz = 20; // Domain depth

// Slope parameters
const double slope_height = 10; // Amplitude of highest (right) point on slope
const double slope_scale = 1e3; // Transition scale of slope


class userControl : public BaseCase {
   public:
      int szx, szz, plotnum, itercount;
      double plot_interval, nextplot;


      DTArray * xgrid, * zgrid, * eta;

      int size_x() const { return szx; }
      int size_y() const { return 1; }
      int size_z() const { return szz; }

      DIMTYPE type_x() const { return NO_SLIP; }
      DIMTYPE type_z() const { return NO_SLIP; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      double get_diffusivity(int t) const {
         return 1e-4;
      }
      double get_visco() const {
         return 1e-4;
      }

      int numActive() const { return 1; }

      double length_x() const { return Lx; }
      double length_y() const { return 1; }
      double length_z() const { return Lz; }

      bool is_mapped() const { return true; }
      void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
         // simple mapping, slope the bottom boundary
         xgrid = alloc_array(szx,1,szz);
         zgrid = alloc_array(szx,1,szz);
         Array<double,1> xx(split_range(szx)), zz(szz);
         // Define numerical Cheby coordinates.  Use the arcsin mapping to
         // stretch the grid towards uniformity
         //xx = cos(ii*M_PI/(szx-1));
         xx = cos(ii*M_PI/(szx-1))/2+0.5;
         zz = cos(ii*M_PI/(szz-1));

         xg = xmin+Lx*xx(ii) + 0*jj+0*kk;
         *xgrid = xg;

         yg = 0;

         zg = -Lz/2 + Lz/2*zz(kk) + slope_height*(1+erf((xg(ii,jj,kk)-1e3)/slope_scale))*(0.5-0.5*zz(kk));
//         zg = -Lz+Lz*zz(kk) + hill_size/2*(1-zz(kk))*(pow(cosh((xg(ii,jj,kk)-hill_loc)/hill_width),-2));
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
            write_array(u,"u",plotnum);
            write_array(w,"w",plotnum);
            write_array(*(tracer[0]),"t",plotnum);
            plotted = true;
            write_array(pressure,"pres",plotnum);
         }
         itercount++;
         if (!(itercount % 1) || plotted) {
            /* Print out some diagnostic information */
            double maxt = pvmax(*tracer[0]);
            double mint = pvmin(*tracer[0]);
            double mu = psmax(max(abs(u))),
                   mw = psmax(max(abs(w))),
                   tdiff = maxt - mint;
            if (master()) {
               fprintf(stderr,"%f (%d): (%.2g, %.2g), (%.2g)\n",
                  time, itercount, mu, mw, tdiff);
            }
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         //do_mapping(u,v,w);
         v = 0;
         u = 0;
         w = 0;
         write_reader(u,"u",true);
         write_reader(w,"w",true);
         write_reader(w,"pres",true);
         write_array(u,"u",0);
         write_array(w,"w",0);
      }
      void init_tracer(int t_num, DTArray & t) {
         
         // The base stratification is pic_strength/2*tanh((zz(kk)-pic_depth)/pic_width)
         // but the grid is mapped between the i/k indices.

//         rho = -pic_strength/2*tanh((1/pic_width) *
//                  ((*zgrid)(ii,jj,kk) - pic_depth - cos_height*sin((*xgrid)(ii,jj,kk)*M_PI/Lx/2)));

         t = bkg_t + lin_t*(Lz + (*zgrid)(ii,jj,kk) + seiche_height*((*xgrid)(ii,jj,kk)-0.5*(Lx+xmin))/Lx)/Lz + 
            pic_t*(0.5+0.5*tanh(((*zgrid)(ii,jj,kk) - pic_H + seiche_height*((*xgrid)(ii,jj,kk)-0.5*(Lx+xmin))/Lx)/pic_W));

                     
         write_array(t,"t",0);
         write_reader(t,"t",true);
      }
      
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         /* Simple gravity-based body forcing, with gravity pointing
            downwards in the y-direction. */
         u_f = 0; v_f = 0;
         w_f = -g*(eqn_of_state_t(*tracers[0]))/rho_0;
      }
      void tracer_forcing(double t, DTArray & u, DTArray & v,
            DTArray & w, vector<DTArray *> & tracers_f) {
         /* Forcing on the tracers themselves.  Since rho is a passive density,
            there is none. */
         *tracers_f[0] = 0;
      }

      userControl() : 
         szx(4096), szz(64),
         //szx(512), szz(128),
//         szx(48), szz(16),
//         szx(24), szz(8),
         plotnum(0), plot_interval(60), nextplot(plot_interval),
         itercount(0) {
         }
};

int main() {
   MPI_Init(0,0);
   userControl mycode;
   FluidEvolve<userControl> slosh(&mycode);
   slosh.initialize();
   slosh.do_run(7200);
   MPI_Finalize();
   return 0;
}
