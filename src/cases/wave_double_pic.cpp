/* Solitary wave crash case with a double picnocline, re: GUPS (Guppy) model */

#include "../BaseCase.hpp"
#include "../TArray.hpp"
#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include "../Science.hpp"
#include "../Par_util.hpp"
#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <blitz/array.h>
#include <vector>

using namespace std;
using namespace TArray;
using namespace NSIntegrator;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

class userControl : public BaseCase {
   public:
      int szx, szz, plotnum, itercount;
      double Lx, Lz, plot_interval, nextplot;

      /* Variables to control stratification */
      double pic_width1, pic_width2, height;
      double width, amplitude;
      /* (reduced) gravity */
      double g;

      /* Depth array */
      Array<double,1> zz;

      int size_x() const { return szx; }
      int size_y() const { return 1; }
      int size_z() const { return szz; }

//      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      /* One active tracer -- density */
      int numActive() const { return 1; }

      double length_x() const { return Lx; }
      double length_y() const { return 1; }
      double length_z() const { return Lz; }

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
         static double maxdiag = 0; // preserve "last calculated" maximum diagnostic
         if (time > (nextplot - 1e-8*fabs(nextplot))) {
           blitz::Array<double,3> diag(split_range(szx),
                  blitz::Range(1,1),blitz::Range(1,1));
            nextplot = nextplot + plot_interval;
            plotnum++;
            diag = overturning_2d(*tracer[0],zz);
            write_array(diag,"diagnostic",plotnum);
            maxdiag = pvmax(diag);
            if (master()) fprintf(stdout,"*");
            if (!(plotnum % 10)) {
               write_array(u,"u_output",plotnum);
               write_array(w,"w_output",plotnum);
               write_array(*(tracer[0]),"rho_output",plotnum);
               plotted = true;
            }
         }
         itercount++;
         if (!(itercount % 25) || plotted) {
            /* Print out some diagnostic information */
            double mu = psmax(max(abs(u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w))),
                   mt = psmax(max(abs(*tracer[0]))),
                   ke = pssum(sum(u*u+v*v+w*w))/(szx*szz);
            if (master())
            fprintf(stdout,"%f: (%.2g, %.2g, %.2g: %.2f), (%.2g) - (%.2g)\n",
                  time, mu, mv, mw, mt, ke, maxdiag);
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         u = 0; v=0; w=0;
         write_reader(u,"u_output",true);
         write_reader(w,"v_output",true);
      }

      void init_tracer(int t_num, DTArray & rho) {
         /* Initialize density to a "tilted" interface, as if an at-rest-but-
            tilted tank were suddenly returned to the horizontal */
         assert(t_num == 0);
         Array<double,1> xx(split_range(szx)), 
            interface(split_range(szx));
         xx = Lx*(ii+0.5)/szx;
         zz = -Lz/2 + Lz*(ii+0.5)/szz;
         Array<double,3> grid(alloc_lbound(szx,1,szz),
                              alloc_extent(szx,1,szz),
                              alloc_storage(szx,1,szz));
         grid = xx(ii) + 0*jj + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = 0*ii + 0*jj + zz(kk);
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);
         
         interface = amplitude/pow(cosh((xx(ii))/width),2);
         double height1 = height ; // Height for weak picnocline
         double height2 = height ; // Height for strong picnocline
/*
         cout << rho.lbound() << endl;
         cout << rho.extent() << endl;
         cout << zz.lbound() << endl;
         cout << zz.extent() << endl;
         cout << interface.lbound() << endl;
         cout << interface.extent() << endl;
         exit(1);*/

         rho = 0.5*(1 - 0.9*tanh(1/pic_width1*(zz(kk)-height2- interface(ii))) -
                        0.1*tanh(1/pic_width2*(zz(kk)-height1- interface(ii))))
               + 0*kk;
         /*rho = 0.5 - 
            0.5*tanh(1/pic_width*(yy(jj) - height -
                     amplitude/pow(cosh(xx(ii)/width),2))) + 0*kk;
         */
         write_array(rho,"rho_output",0); write_reader(rho,"rho_output",true); 
//         write_array(overturning_2d(rho,zz),"diagnostic",0);
         write_reader(overturning_2d(rho,zz),"diagnostic",true);
      }

      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         /* Simple gravity-based body forcing, with gravity pointing
            downwards in the y-direction. */
         u_f = 0; v_f = 0;
         w_f = -g*(*tracers[0]);
      }
      void tracer_forcing(double t, const DTArray & u, const DTArray & v,
            const DTArray & w, vector<DTArray *> & tracers_f) {
         /* Forcing on the tracers themselves.  Since rho is a passive density,
            there is none. */
         *tracers_f[0] = 0;
      }

      userControl() : 
         szx(4096), szz(2048), Lx(40), Lz(1),
         plotnum(0), plot_interval(0.025), nextplot(plot_interval),
         itercount(0), g(.2), 
         height(-0.25), pic_width1(0.05), pic_width2(0.005),
         width(3), amplitude(0.35), zz(szz) {
         }
};

int main() {
   MPI_Init(0,0);
   userControl mycode;
   FluidEvolve<userControl> slosh(&mycode);
   slosh.initialize();
   slosh.do_run(600);
   MPI_Finalize();
   return 0;
}
