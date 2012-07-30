/* Copy of test case used in MATLAB code, collision of an initially
   shielded vortex dipole with a no-slip boundary */

#include "../Par_util.hpp"
#include <mpi.h>
#include "../BaseCase.hpp"
#include "../TArray.hpp"
#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include <stdio.h>
#include <random/normal.h>
#include <vector>

using namespace std;
using namespace TArrayn;
using namespace NSIntegrator;
using namespace ranlib;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

#define D1_X -.1
#define D1_Z 0
#define D2_X .1
#define D2_Z 0
#define RAD2 0.01

class userControl : public BaseCase {
   public:
      int szx, szy, szz, plotnum, itercount, lastplot;
      double plot_interval, nextplot;

      DIMTYPE type_x() const { return FREE_SLIP; }
      DIMTYPE type_y() const { return FREE_SLIP; }
      DIMTYPE type_z() const { return NO_SLIP; }

      double get_visco() const {
         return 1.0/4000;
      }
      double length_x() const { return 2; }
      double length_y() const { return 2; }
      double length_z() const { return 2; }

      int size_x() const { return szx; }
      int size_y() const { return szy; }
      int size_z() const { return szz; }

      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            if (master()) fprintf(stderr,"Tiny timestep, aborting\n");
            return -1;
         } else if (itercount < 100 && intime > .01) {
            intime = .01;
         }
         if (intime > plot_interval) intime = plot_interval;
         double until_plot = nextplot - now;
         if (until_plot < 0) {
            /* We've passed the proper plot time */
            return intime / 100;
         }
         double steps = ceil(until_plot / intime);
         double real_until_plot = steps*intime;
         if (fabs(until_plot - real_until_plot) < 1e-5*plot_interval) {
            return intime;
         } else {
            return (until_plot / steps);
         }
      }

      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer) {
         /* Write out velocities */
         bool plotted = false;
         itercount++;
         if ((time - nextplot) > -1e-5*plot_interval) {
            plotted = true;
            nextplot += plot_interval;
            if (master()) fprintf(stdout,"*");
            plotnum++;
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(w,"w_output",plotnum);
            lastplot = itercount;
         }
         if ((itercount - lastplot)%100 == 0 || plotted) {
            double mu = psmax(max(abs(u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w)));
            if (master())
               fprintf(stdout,"%f [%d] (%.4g, %.4g, %.4g)\n",
                     time, itercount, mu, mv, mw);
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         Array<double,1> xx(split_range(szx)), yy(szy), zz(szz);
         xx = -(length_x()/2) + length_x()*(ii+0.5)/szx;
         yy = -(length_y()/2) + length_y()*(ii+0.5)/szy;
         zz = -(length_z()/2)*cos(M_PI*ii/(szz-1));
         Array<double,3> grid(alloc_lbound(szx,szy,szz),
                              alloc_extent(szx,szy,szz),
                              alloc_storage(szx,szy,szz));
         u = 160*(
            +(zz(kk)-D1_Z)*exp(-(pow(xx(ii)-D1_X,2)+pow(zz(kk)-D1_Z,2))/RAD2) -
             (zz(kk)-D2_Z)*exp(-(pow(xx(ii)-D2_X,2)+pow(zz(kk)-D2_Z,2))/RAD2));
         w = 160*(
            -(xx(ii)-D1_X)*exp(-(pow(xx(ii)-D1_X,2)+pow(zz(kk)-D1_Z,2))/RAD2) +
             (xx(ii)-D2_X)*exp(-(pow(xx(ii)-D2_X,2)+pow(zz(kk)-D2_Z,2))/RAD2));
         v = 0;
         write_array(u,"u_output",0);
         write_reader(u,"u_output",true);
         write_array(v,"v_output",0);
         write_reader(v,"v_output",true);
         write_array(w,"w_output",0);
         write_reader(w,"w_output",true);
         grid = xx(ii) + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = yy(jj) + 0*kk;
         write_array(grid,"ygrid"); write_reader(grid,"ygrid",false);
         grid = zz(kk);
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);
      }
      void passive_forcing(double t, const DTArray & u, DTArray & u_f, 
            const DTArray & v, DTArray & v_f, 
            const DTArray & w, DTArray & w_f) {
         u_f = 0;
         v_f = 0;
         w_f = 0;
      }
      userControl():
         szx(1024), szy(1), szz(1025),
         plotnum(0), plot_interval(.01), nextplot(plot_interval),
         itercount(0) {
         }
};

int main() {
   MPI_Init(0,0);
   userControl mycode;
   FluidEvolve<userControl> ppois(&mycode);
   ppois.initialize();
   ppois.do_run(2);
   MPI_Finalize();
   return 0;
}
         
            
