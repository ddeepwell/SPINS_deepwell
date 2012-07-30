/* Copy of test case used in MATLAB code, collision of an initially
   shielded vortex dipole with a no-slip boundary */

#include "../Par_util.hpp"
#include <mpi.h>
#include "../BaseCase.hpp"
#include "../TArray.hpp"
#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <random/normal.h>
#include <vector>
#include "../Science.hpp"

using namespace std;
using namespace TArrayn;
using namespace NSIntegrator;
using namespace ranlib;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

#define P0 1e3 // Base density

// Following are from Smyth
#define H0 0.092 
#define DU 0.00332
#define DP 1.21e-3
#define LX 2.62
#define LY 0.65
#define LZ 0.88



double times_record[100];
double max_dens[100], min_dens[100], ke[100];
int myrank = -1;

class userControl : public BaseCase {
   public:
      int szx, szy, szz, plotnum, itercount, lastplot, last_writeout;
      double plot_interval, nextplot;

      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_y() const { return PERIODIC; }
      DIMTYPE type_z() const { return FREE_SLIP; }

      double get_visco() const {
         return 1e-6;
      }
      double get_diffusivity(int t) const {
         return 0e-6;
      }
      double length_x() const { return LX; }
      double length_y() const { return LY; }
      double length_z() const { return LZ; }

      int size_x() const { return szx; }
      int size_y() const { return szy; }
      int size_z() const { return szz; }

      int numActive() const { return 1; }

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
            vector<DTArray *> tracer, DTArray & pressure) {
         /* Write out velocities */
         bool plotted = false;
         itercount++;

         times_record[itercount-last_writeout-1] = time;


         max_dens[itercount-last_writeout-1] = pvmax(*tracer[0]);
         min_dens[itercount-last_writeout-1] = pvmin(*tracer[0]);
         ke[itercount-last_writeout-1] = 
            pssum(sum((*get_quad_x())(ii)*(*get_quad_y())(jj)*
                     (*get_quad_z())(kk)*(pow(u(ii,jj,kk),2)+
                                          pow(v(ii,jj,kk),2)+
                                          pow(w(ii,jj,kk),2))));

         
         if ((time - nextplot) > -1e-5*plot_interval) {
            plotted = true;
            nextplot += plot_interval;
            if (master()) fprintf(stdout,"*");
            plotnum++;
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(w,"w_output",plotnum);
            write_array(*tracer[0],"rho_output",plotnum);
            lastplot = itercount;
            if (master()) {
               FILE * plottimes = fopen("plot_times.txt","a");
               assert(plottimes);
               fprintf(plottimes,"%.10g\n",time);
               fclose(plottimes);
            }
         }
         if ((itercount - lastplot)%10 == 0 || plotted) {
            double mu = psmax(max(abs(u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w)));
            if (master())
               fprintf(stdout,"%f [%d] (%.4g, %.4g, %.4g) -- (%g, %g)\n",
                     time, itercount, mu, mv, mw,
                     max_dens[itercount-last_writeout-1]-
                      min_dens[itercount-last_writeout-1]-2*DP,
                     ke[itercount-last_writeout-1]);
            if (master()) {
               FILE * en_record = fopen("energy_record.txt","a");
               assert(en_record);
               for (int i = 0; i < (itercount-last_writeout); i++) {
                  fprintf(en_record,"%g %.12e %.12e %.12e %.12e\n",
                        times_record[i], ke[i],
                        max_dens[i],min_dens[i],max_dens[i]-min_dens[i]-2*DP);                      }
               fclose(en_record);
            }
            last_writeout = itercount;
         }
      }

      void init_tracers(vector<DTArray *> & tracers) {
         assert(tracers.size() == 1);
         Array<double,1> xx(split_range(szx)), zz(szz);
         zz = -(length_z()/2) + length_z()*(ii+0.5)/szz;
         xx = -(length_x()/2) + length_x()*(ii+0.5)/szx;

         *tracers[0] = -DP*tanh((zz(kk)+
                                 LZ*.001*cos(xx(ii)*2*M_PI/LX+M_PI/2) +
                                 LZ*.01*sin(xx(ii)*4*M_PI/LX))/H0);
         write_array(*tracers[0],"rho_output",0);
         write_reader(*tracers[0],"rho_output",true);
         
      }
      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         Array<double,1> xx(split_range(szx)), yy(szy), zz(szz);
         xx = -(length_x()/2) + length_x()*(ii+0.5)/szx;
         yy = -(length_y()/2) + length_y()*(ii+0.5)/szy;
         zz = -(length_z()/2) + length_z()*(ii+0.5)/szz;

         Array<double,3> grid(alloc_lbound(szx,szy,szz),
                              alloc_extent(szx,szy,szz),
                              alloc_storage(szx,szy,szz));
         u = DU*tanh(zz(kk)/H0);
         v = 0;
         w = 0;
         /* Add random initial perturbation */
         int myrank;
         MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
         /* Add random noise about 3.5 orders of magnitude below dipole */
         Normal<double> rnd(0,1);
         rnd.seed(myrank);
         for (int i = u.lbound(firstDim); i<= u.ubound(firstDim); i++) {
            for (int j = u.lbound(secondDim); j<= u.ubound(secondDim); j++) {
               for (int k = u.lbound(thirdDim); k<= u.ubound(thirdDim); k++) {
                  u(i,j,k) += DU*1e-3*rnd.random();
                  v(i,j,k) += DU*1e-3*rnd.random();
                  w(i,j,k) += DU*1e-3*rnd.random();
               }
            }
         }
         /* This noise is neither incompressible nor satisfying of the boundary
            conditions.  The code will make it do both after the first timestep.*/
         if (szy == 1) {
            v = 0;
         }
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
      void vel_forcing(double t, DTArray & u_f, DTArray & v_f, DTArray & w_f,
            vector<DTArray *> & tracers) {
         u_f = 0; v_f = 0;
         w_f = -9.81/P0*(*tracers[0]);
      }
      void tracer_forcing(double t, const DTArray & u, const DTArray & v,
            const DTArray & w, vector<DTArray *> & tracers_f) {
         *tracers_f[0]=0;
      }
      userControl(int s1, int s2, int s3):
         szx(s1), szy(s2), szz(s3),
         plotnum(0), plot_interval(100), nextplot(plot_interval),
         lastplot(0),
         itercount(0),last_writeout(0) {
            compute_quadweights(szx,szy,szz,
                  length_x(),length_y(),length_z(),
                  type_x(),type_y(),type_z());
            if (master()) {
               printf("Using array size %dx%dx%d\n",s1,s2,s3);
            }
         }
};

int main(int argc, char ** argv) {
   /* Set filter parameters */
   f_strength = -.25;
   f_order = 2;
   f_cutoff = 0.8;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   userControl mycode(512,1,256);
   FluidEvolve<BaseCase> ppois(&mycode);
   ppois.initialize();
   ppois.do_run(12000);
   MPI_Finalize();
   return 0;
}
         
            
