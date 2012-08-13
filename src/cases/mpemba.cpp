/* Mpemba effect -- hot water freezing before cold water */
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

const double g = 1;
const double rho0 = 1;
double drho = 1;


const double Lx = 2;
const double Ly = 1;
const double Lz = 1;

double nu = 1e-3;
double kappa = 1e-3;

double prandtl = 1;

class userControl : public BaseCase {
   public:
      int szx, szz, plotnum, itercount;
      double plot_interval, nextplot;

      Array<double,1> xx, zz ;

      int size_x() const { return szx; }
      int size_y() const { return 1; }
      int size_z() const { return szz; }

      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_z() const { return NO_SLIP; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      double get_diffusivity(int t) const {
         return kappa;
      }
      double get_visco() const {
         return nu;
      }

      int numActive() const { return 1; }

      double length_x() const { return Lx; }
      double length_y() const { return Lz; }
      double length_z() const { return Ly; }

      bool is_mapped() const { return false; }

      /* This (and writeout for analysis) sould probably be abstracted into
         another mixin class */
      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            /* Something's gone wrong, so abort */
            fprintf(stderr,"Tiny timestep returned, aborting\n");
            return -1;
         } else if (intime > .001) {
            /* Cap the maximum timestep size */
            intime = .001; // And process to make sure we land on a plot time
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
         }
         itercount++;
//         if (!(itercount % 25) || plotted) {
            /* Print out some diagnostic information */
            double mu = psmax(max(abs(u))),
                   mw = psmax(max(abs(w)));
            double avg_t = pssum(
                  sum((*(get_quad_x()))(ii) *
                      (*(get_quad_y()))(jj) *
                      (*(get_quad_z()))(kk) *
                      (*(tracer[0]))(ii,jj,kk) /
                      (Lx*Ly*Lz)));
            double ke = pssum(
                  sum((*(get_quad_x()))(ii) *
                      (*(get_quad_y()))(jj) *
                      (*(get_quad_z()))(kk) *
                      (0.5*(u*u+v*v*w*w)) /
                      (Lx*Ly*Lz)));
            double pe = 0;
            if (master()) {
               fprintf(stderr,"%f: (%.2g, %.2g), (%.4g), (%.4g %.4g)\n",
                  time, mu, mw, avg_t,ke,pe);
               FILE* fp = fopen("energetics.txt","a+");
               fprintf(fp,"%.10g %.10g %.10g %.10g\n",time,avg_t,ke,pe);
               fclose(fp);
            }
//         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
//         u = 0; v=0; w=0;
         v = 0; w = 0;
//         u = sin(M_PI*zz(kk)/Lz);
         u = drho/nu*(Lz*Lz/(M_PI*M_PI)*sin(M_PI*zz(kk)/Lz) -
            12*Lz*Lz/(M_PI*M_PI*M_PI)*(zz(kk))*(Lz-zz(kk)));
         fprintf(stderr,"%g\n",u(64,0,64));
//         MPI_Finalize(); exit(1);
         write_reader(u,"u",true);
         write_reader(w,"w",true);
         write_array(u,"u",0);
         write_array(w,"w",0);
      }
      void init_tracers(vector<DTArray *> & tracers) {
         /* Initialize heat, salt tracers */
         DTArray & heat = *tracers[0];

         Array<double,3> grid(alloc_lbound(szx,1,szz),
                              alloc_extent(szx,1,szz),
                              alloc_storage(szx,1,szz));


         Normal<double> rnd(0,1); // Random generator

         for (int i = heat.lbound(firstDim);
               i <= heat.ubound(firstDim); i++) {
            rnd.seed(i);
            for (int j = heat.lbound(secondDim);
                  j <= heat.ubound(secondDim); j++) {
               for (int k = heat.lbound(thirdDim);
                     k <= heat.ubound(thirdDim); k++) {
                  heat(i,j,k) = rho0*(drho*sin(M_PI*zz(k)/Lz)+1e-3*rnd.random());
               }
            }
         }
         write_array(heat,"t",0); 
         write_reader(heat,"t",true);

         grid = xx(ii)+0*kk;
         write_array(grid,"xgrid");
         write_reader(grid,"xgrid",false);

         grid = zz(kk);
         write_array(grid,"zgrid");
         write_reader(grid,"zgrid",false);
      }
      
      // Forcing must be done generally, since both rotation and density are
      // involved
      void forcing(double t, DTArray & u, DTArray & u_f, 
            DTArray & v, DTArray & v_f, DTArray & w,
            DTArray & w_f, vector<DTArray *> & tracers,
            vector<DTArray *> & tracers_f) {
         u_f = g*(*(tracers[0]))/rho0;
         // Add correction force to ensure no net left/right flow
         double duf = 0;
         duf = pssum(
                  sum((*(get_quad_x()))(ii) *
                      (*(get_quad_y()))(jj) *
                      (*(get_quad_z()))(kk) *
                      (u_f(ii,jj,kk)) /
                      (Lx*Ly*Lz)));
         //fprintf(stderr,"%g + ",duf);
         u_f = u_f - duf;
         // Also add in correction to remove any net left/right flow
         // that is already present
         duf = 250* // timescale
            pssum(
                  sum((*(get_quad_x()))(ii) *
                      (*(get_quad_y()))(jj) *
                      (*(get_quad_z()))(kk) *
                      (u(ii,jj,kk)) /
                      (Lx*Ly*Lz)));
         u_f = u_f - duf;
         //fprintf(stderr,"%g\n",duf);

         v_f = 0;//ROT_F*u;
         w_f = 0;//-g*eqn_of_state_t(*(tracers[0]))/rho0;
         // And since the temp content is expressed as total content rather
         // than perturbation, no forcing is necessary.
         *tracers_f[0] = 0;
      }

      userControl() : 
         szx(128), szz(128),
         plotnum(0), plot_interval(.1), nextplot(plot_interval),
         itercount(0),
         xx(split_range(szx)), zz(szz)
      {
         compute_quadweights(szx,1,szz,
               length_x(), length_y(), length_z(),
               type_x(), type_y(), type_z()
               );
         //xx = Lx/2*cos(M_PI*ii/(szx-1));
         xx = (0.5+ii)/szx*Lx - Lx/2;
         zz = Lz/2*(1+cos(M_PI*ii/(szz-1)));

         }
};

int main(int argc, char ** argv) {
   zero_tracer_boundary = true;
   MPI_Init(&argc,&argv);
   if (argc > 1) {
      nu = atof(argv[1]);
   }
   if (argc > 2) {
      prandtl = atof(argv[2]);
   }
   if (nu <= 0) nu=1e-3;
   if (prandtl <= 0) prandtl = 1;
   kappa = nu*prandtl;
   double finaltime = log(10)/M_PI/M_PI/kappa;
   if (master()) {
      fprintf(stderr,"Mpemba Effect: cooling of hot liquid under free convection\n");
      fprintf(stderr,"Grashof number: %.3e\nPrandtl number: %.3g\n",1/nu/nu,kappa/nu);
      fprintf(stderr,"Runnig to 1/10-time %g\n",finaltime);
   }
   userControl mycode;
   FluidEvolve<userControl> slosh(&mycode);
   slosh.initialize();
   slosh.do_run(finaltime);
   MPI_Finalize();
   return 0;
}
