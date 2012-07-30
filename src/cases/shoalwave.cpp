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
const double pic_depth = -.2; // picnocline depth
const double pic_strength = .01; // strength
const double pic_width = .01;

// Parameters for the initial density disturbance
const double sech_height = .1;
const double sech_width = 1;
const double sech_loc = -3;

const double Lx = 9; // domain half-length
const double Lz = .5; // domain half-height
const double hill_size = 0.4; // Hill size
const double hill_loc = 0.6; // hill location, normalized to [-1,1]a
const double hill_width = 0.25;
const double right_slope = 0; // Slope of right boundary (inverse)
   // Salt diffusivity is O(10^-9) ~= 0 

/*   So, to more than balance out a 5 K temperature change (delta_rho of
   1.25g, we want a weak, weak salt stratification of about 1/50 mol/L 
   */
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
         return 1e-5;
      }
      double get_visco() const {
         return 1e-5;
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
         xx = cos(ii*M_PI/(szx-1));
         zz = cos(ii*M_PI/(szz-1));

         xg = Lx*xx(ii) + (1+xx(ii))*zz(kk)*right_slope;
         *xgrid = xg;

         yg = 0;

         zg = Lz*zz(kk) + hill_size/4*(1-zz(kk))*(1+tanh((xx(ii)-hill_loc)/hill_width));
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
         } else if (itercount < 10 && intime > 1e-3) {
            /* Cap the maximum timestep size */
            intime = 1e-3; // And process to make sure we land on a plot time
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
         v = 0;
         // linear long wave speed
         double longwave = sqrt(g*pic_strength*(Lz-pic_depth)*(Lz+pic_depth));

         // Some parameters for the z-dependence of eta
         double chmin = log(cosh((-Lz - pic_depth)/pic_width)),
                chmax = log(cosh(( Lz - pic_depth)/pic_width)),
                amp_z = -1 / (chmin*(Lz - pic_depth)/(2*Lz) + chmax*(Lz + pic_depth)/(2*Lz));
         eta = alloc_array(szx,1,szz);

         *eta = sech_height*pow(cosh(((*xgrid)(ii,jj,kk)-sech_loc)/sech_width),-2) *
               amp_z * (log(cosh(((*zgrid)(ii,jj,kk) - pic_depth)/pic_width)) -
                        chmin*(Lz - (*zgrid)(ii,jj,kk))/(2*Lz) -
                        chmax*(Lz + (*zgrid)(ii,jj,kk))/(2*Lz)) ;

         write_array(*eta,"init_eta");
         write_reader(*eta,"init_eta",false);
         // w = -eta_x*longwave
         w = longwave * 2/sech_width * (*eta)(ii,jj,kk) * tanh(((*xgrid)(ii,jj,kk) - sech_loc)/sech_width) ;

         // u = eta_z*longwave
         u = longwave * amp_z * sech_height * pow(cosh( ((*xgrid) - sech_loc)/sech_width ),-2) *
            (tanh(((*zgrid)(ii,jj,kk)-pic_depth)/pic_width)/pic_width +
             chmin / (2*Lz) - chmax / (2*Lz)) ;
         write_reader(u,"u_output",true);
         write_reader(w,"w_output",true);
         write_array(u,"u_output",0);
         write_array(w,"w_output",0);
      }
      void init_tracer(int t_num, DTArray & rho) {
         
         // The base stratification is pic_strength/2*tanh((zz(kk)-pic_depth)/pic_width)
         // but the grid is mapped between the i/k indices.

         // The background stratification is -pic_strength/2*tanh((z-pic_depth)/pic_width)
         // and on top of that we impose an isopycnical displacement for a mode 1
         // wave similar in form to a solitary wave, for a full density field of
         // rho = rho_z(z-eta)

         // Some parameters for the z-dependence of eta
         double chmin = log(cosh((-Lz - pic_depth)/pic_width)),
                chmax = log(cosh(( Lz - pic_depth)/pic_width)),
                amp_z = -1 / (chmin*(Lz - pic_depth)/(2*Lz) + chmax*(Lz + pic_depth)/(2*Lz));

         rho = -pic_strength/2*tanh((1/pic_width)*sech_height *
                  ((*zgrid)(ii,jj,kk) - pic_depth - (*eta)(ii,jj,kk)));
                     
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
      void tracer_forcing(double t, const DTArray & u, const DTArray & v,
            const DTArray & w, vector<DTArray *> & tracers_f) {
         /* Forcing on the tracers themselves.  Since rho is a passive density,
            there is none. */
         *tracers_f[0] = 0;
      }

      userControl() : 
         szx(1024), szz(128),
//         szx(128), szz(32),
         plotnum(0), plot_interval(.1), nextplot(plot_interval),
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
