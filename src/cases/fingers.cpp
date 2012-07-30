/* Salt fingers */
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
const double rho_salt = .028; // kg / mol, salt density
const double rho_h = 2.5e-4; // kg / L / K (thermal expansion of water)
const double alpha = 1.3e-7; // -- thermal diffusivity for water
const double beta = 1e-8;
   // Salt diffusivity is O(10^-9) ~= 0 

/*   So, to more than balance out a 5 K temperature change (delta_rho of
   1.25g, we want a weak, weak salt stratification of about 1/50 mol/L 
   */
class userControl : public BaseCase {
   public:
      int szx, szy, plotnum, itercount;
      double Lx, Ly, plot_interval, nextplot;

      /* Stratified layer, thickness */
      double height, thickness;
      /* Stratified variables */
      double delta_h, delta_s;
      
      static const int SALT = 0;
      static const int HEAT = 1;

      int size_x() const { return szx; }
      int size_y() const { return 1; }
      int size_z() const { return szy; }

//      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      double get_diffusivity(int t) const {
         switch (t) {
            case SALT:
               return beta;
            case HEAT:
               return alpha;
            default:
               abort();
         }
      }
      double get_visco() const {
         return 1e-6;
      }

      int numActive() const { return 2; }

      double length_x() const { return Lx; }
      double length_y() const { return 1; }
      double length_z() const { return Ly; }

      /* This (and writeout for analysis) sould probably be abstracted into
         another mixin class */
      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            /* Something's gone wrong, so abort */
            fprintf(stderr,"Tiny timestep returned, aborting\n");
            return -1;
         } else if (itercount < 100 && intime > .1) {
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
            if (master()) fprintf(stderr,"*");
            plotnum++;
            write_array(u,"u_output",plotnum);
            write_array(w,"w_output",plotnum);
            write_array(*(tracer[HEAT]),"h_output",plotnum);
            write_array(*(tracer[SALT]),"s_output",plotnum);
            plotted = true;
         }
         itercount++;
         if (!(itercount % 25) || plotted) {
            /* Print out some diagnostic information */
            double mu = psmax(max(abs(u))),
                   mw = psmax(max(abs(w))),
                   ke = pssum(sum(u*u+w*w))/szx/szy*Lx*Ly;
            if (master()) {
               fprintf(stderr,"%f: (%.2g, %.2g), (%.2g)\n",
                  time, mu, mw, ke);
            }
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         u = 0; v=0; w=0;
         write_reader(u,"u_output",true);
         write_reader(w,"w_output",true);
         write_array(u,"u_output",0);
         write_array(w,"w_output",0);
      }
      void init_tracers(vector<DTArray *> & tracers) {
         /* Initialize heat, salt tracers */
         DTArray & heat = *tracers[HEAT];
         DTArray & salt = *tracers[SALT];

         Array<double,1> xx(split_range(szx)), zz(szy), 
            interface(split_range(szx));
         xx = -Lx/2 + Lx*(ii+0.5)/szx;
         zz = -Ly/2 + Ly*(ii+0.5)/szy;
         Array<double,3> grid(alloc_lbound(szx,1,szy),
                              alloc_extent(szx,1,szy),
                              alloc_storage(szx,1,szy));
         grid = xx(ii) + 0*jj + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = 0*ii + zz(kk) + 0*jj;
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);

         /* Generate the interface perturbation.  This is essentially low-
            amplitude white noise.  Yes, this isn't "well-resolved" on the
            spectral grid, but there's no need to care.  High frequencies
            will be damped out by the filtering anyway, and this mostly
            exists as a way to impose a perturbation in heat/salt that
            a) matches, and
            b) remains positive throughout */
         Normal<double> rnd(0,sqrt(thickness*1e-6)); // Random generator
         /* Seed the random number generator with the MPI rank */
         int myrank;
         MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
         rnd.seed(myrank);
         for (int i = interface.lbound(firstDim); 
               i <= interface.ubound(firstDim); i++) {
           interface(i) = rnd.random();
         } 
         assert (tracers.size() == 2);
         heat = delta_h*(0.5 + 
               0.5 * tanh((zz(kk)-height-interface(ii))/thickness)) + 0*kk;
         salt = delta_s*(0.5 +
               0.5 * tanh((zz(kk)-height)/thickness)) + 0*kk;

         write_array(heat,"h_output",0); 
         write_reader(heat,"h_output",true);
         write_array(salt,"s_output",0);
         write_reader(salt,"s_output",true);
      }
      
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         /* Simple gravity-based body forcing, with gravity pointing
            downwards in the y-direction. */
         DTArray & heat = *tracers[HEAT];
         DTArray & salt = *tracers[SALT];
         u_f = 0; v_f = 0;
         w_f = -g*(rho_0 + rho_salt*salt - rho_h*heat);
      }
      void tracer_forcing(double t, const DTArray & u, const DTArray & v,
            const DTArray & w, vector<DTArray *> & tracers_f) {
         /* Forcing on the tracers themselves.  Since rho is a passive density,
            there is none. */
         *tracers_f[HEAT] = 0;
         *tracers_f[SALT] = 0;
      }

      userControl() : 
         szx(1024), szy(1024), Lx(.05), Ly(.05),
         plotnum(0), plot_interval(.1), nextplot(plot_interval),
         itercount(0),
         delta_h(5), delta_s(.04), height(0), thickness(Ly/10) {
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
