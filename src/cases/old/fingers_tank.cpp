/* Salt fingers with no-slip walls */
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
const double g = 9.81; // m/s^2, gravitya

const double T_0 = 15; // Temperature of ''cold'' water
const double DT = 25; // Change in temperature to ''hot'' water
const double DS = 5; // Change in salinity
const double rho_0 = 1000; //  kg / L, fresh water at 20 C

const double x_0 = 0.10; // Dam-break location
const double dx = 0.01; // Width of break

const double LX = 0.40;;
const double LY = 0.064;
const double LZ = 0.064;

const int NX = 384;
const int NY = 64;
const int NZ = 96;

const double alpha = 1.7e-7;
const double beta = 1e-8;
const double fudge = 10;


class userControl : public BaseCase {
   public:
      int szx, szy, szz, plotnum, itercount;
      double plot_interval, nextplot;

      
      static const int SALT = 1;
      static const int HEAT = 0;

      DTArray * xgrid, * zgrid, * ygrid;
      Array<double,1> xx,yy,zz;

      int size_x() const { return szx; }
      int size_y() const { return szy; }
      int size_z() const { return szz; }

      DIMTYPE type_x() const { return FREE_SLIP; }
      DIMTYPE type_y() const { return FREE_SLIP; }
      DIMTYPE type_z() const { return NO_SLIP; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      double get_diffusivity(int t) const {
         switch (t) {
            case SALT:
               return fudge*beta;
            case HEAT:
               return fudge*alpha;
            default:
               abort();
         }
      }
      double get_visco() const {
         return fudge*1e-6;
      }

      int numActive() const { return 2; }

      double length_x() const { return LX; }
      double length_y() const { return LY; }
      double length_z() const { return LZ; }

      bool is_mapped() const { return false; }


      /* This (and writeout for analysis) sould probably be abstracted into
         another mixin class */
      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            /* Something's gone wrong, so abort */
            fprintf(stderr,"Tiny timestep returned, aborting\n");
            return -1;
         } else if (intime > 5e-2) {
            /* Cap the maximum timestep size */
            intime = 5e-2; // And process to make sure we land on a plot time
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
            write_array(v,"v",plotnum);
            write_array(w,"w",plotnum);
            write_array(*(tracer[HEAT]),"t",plotnum);
            write_array(*(tracer[SALT]),"s",plotnum);
            write_array(pressure,"pp",plotnum);
            plotted = true;
         }
         itercount++;
//         if (!(itercount % 25) || plotted) {
            /* Print out some diagnostic information */
            double mu = psmax(max(abs(u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w))),
                   deltat = psmax(max(*(tracer[HEAT])))-psmin(min(*(tracer[HEAT]))),
                   deltas = psmax(max(*(tracer[SALT])))-psmin(min(*(tracer[SALT])));
            
            if (master()) {
               fprintf(stderr,"%f: (%.2g, %.2g, %.2g), (%.3g,%3g)\n",
                  time, mu, mv, mw, deltat, deltas);
            }
//         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         u = 0; v=0; w=0;
         write_reader(u,"u",true);
         write_reader(w,"w",true);
         write_reader(v,"v",true);
         write_reader(v,"pp",true);
         write_array(u,"u",0);
         write_array(w,"w",0);
         write_array(v,"v",0);
      }
      void init_tracers(vector<DTArray *> & tracers) {
         /* Initialize heat, salt tracers */
         DTArray & heat = *tracers[HEAT];
         DTArray & salt = *tracers[SALT];

         // Repurpose arrays for grid writing
         heat = xx(ii)+0*jj+0*kk;
         write_array(heat,"xgrid");
         write_reader(heat,"xgrid",false);

         heat = 0*ii + yy(jj) + 0*kk;
         write_array(heat,"ygrid");
         write_reader(heat,"ygrid",false);

         heat = 0*ii + 0*jj + zz(kk);
         write_array(heat,"zgrid");
         write_reader(heat,"zgrid",false);


         //assert (tracers.size() == 2);
         heat = T_0 + DT*(0.5+0.5*tanh((x_0 - xx(ii))/dx));
         salt =  0 + DS*(0.5+0.5*tanh((x_0 - xx(ii))/dx));
         // Add small perturbation to T
         Normal<double> rnd(0,1); // Random generator
         for (int i = heat.lbound(firstDim);
               i <= heat.ubound(firstDim);i++) {
            rnd.seed(i);
            for (int j = heat.lbound(secondDim);
                  j <= heat.ubound(secondDim);j++) {
               for (int k = heat.lbound(thirdDim);
                     k <= heat.ubound(thirdDim);k++) {
                  heat(i,j,k) +=  1e-3*DT*rnd.random();
               }
            }
         }
         

         write_array(heat,"t",0); 
         write_reader(heat,"t",true);
         write_array(salt,"s",0);
         write_reader(salt,"s",true);
      }
      
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         /* Simple gravity-based body forcing, with gravity pointing
            downwards in the y-direction. */
         DTArray & heat = *tracers[HEAT];
         DTArray & salt = *tracers[SALT];
         u_f = 0; v_f = 0;
         w_f = -g*eqn_of_state_t(heat)/rho_0;
      }
      void tracer_forcing(double t, DTArray & u, DTArray & v,
            DTArray & w, vector<DTArray *> & tracers_f) {
         /* Forcing on the tracers themselves.  Since rho is a passive density,
            there is none. */
         *tracers_f[HEAT] = 0;
         *tracers_f[SALT] = 0;
      }

      userControl() : 
         szx(NX), szy(NY), szz(NZ),
         plotnum(0), plot_interval(1), nextplot(plot_interval),
         itercount(0),
         xx(split_range(NX)), yy(NY), zz(NZ)
         {
            xx = LX*(0.5+ii)/NX;
            zz = LZ*(0.5-0.5*cos(ii*M_PI/(NZ-1)));
            yy = LY*(0.5+ii)/NY;
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
