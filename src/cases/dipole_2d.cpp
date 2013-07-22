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

#define D1_X -.1
#define D1_Z 0
#define D2_X .1
#define D2_Z 0
#define RAD2 0.01

double times_record[100], ke_record[100], enst_record[100], ke2d_record[100];
int myrank = -1;

class userControl : public BaseCase {
   public:
      int szx, szy, szz, plotnum, itercount, lastplot, last_writeout;
      double plot_interval, nextplot;

      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_y() const { return PERIODIC; }
      DIMTYPE type_z() const { return NO_SLIP; }

      double get_visco() const {
         return 1.0/1250;
      }
      double length_x() const { return 2; }
      double length_y() const { return 0.4; }
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
            vector<DTArray *> tracer, DTArray & pressure) {
         /* Write out velocities */
         bool plotted = false;
         itercount++;
         DTArray * vor_x, * vor_y, *vor_z;
         vorticity(u,v,w,vor_x,vor_y,vor_z,length_x(),length_y(),length_z(),
            size_x(),size_y(),size_z(),type_x(),type_y(),type_z());

         double enst, ke, ke2d;
         enst = enst_record[itercount-last_writeout-1] = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)*
                  (pow((*vor_x)(ii,jj,kk),2) + pow((*vor_y)(ii,jj,kk),2) +
                   pow((*vor_z)(ii,jj,kk),2))));
         ke = ke_record[itercount-last_writeout-1] = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)*
                  (pow(u(ii,jj,kk),2)+pow(v(ii,jj,kk),2)+pow(w(ii,jj,kk),2))));
         ke2d = ke2d_record[itercount-last_writeout-1] = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_z())(jj)*
                  (pow(mean(u(ii,kk,jj),kk),2)+
                   pow(mean(v(ii,kk,jj),kk),2)+
                   pow(mean(w(ii,kk,jj),kk),2))))*(length_y());

         times_record[itercount-last_writeout-1] = time;

         
         if ((time - nextplot) > -1e-5*plot_interval) {
            plotted = true;
            nextplot += plot_interval;
            if (master()) fprintf(stdout,"*");
            plotnum++;
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(w,"w_output",plotnum);
            lastplot = itercount;
            if (master()) {
               FILE * plottimes = fopen("plot_times.txt","a");
               assert(plottimes);
               fprintf(plottimes,"%.10g\n",time);
               fclose(plottimes);
            }
         }
         if ((itercount - lastplot)%100 == 0 || plotted) {
            double mu = psmax(max(abs(u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w)));
            if (master())
               fprintf(stdout,"%f [%d] (%.4g, %.4g, %.4g) -- (%g, %g, %g)\n",
                     time, itercount, mu, mv, mw,enst,ke,ke2d);
            if (master()) {
               FILE * en_record = fopen("energy_record.txt","a");
               assert(en_record);
               for (int i = 0; i < (itercount-last_writeout); i++) {
                  fprintf(en_record,"%g %g %g %g\n",times_record[i],
                        ke_record[i],enst_record[i],ke2d_record[i]);
               }
               fclose(en_record);
            }
            last_writeout = itercount;
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
         /* Vortex strength is set according to Clercx(2006) and Kramer(2007),
            which normalizes initial KE to 2 and initial enstrophy to 800 */
         u = 299.5284/2*(
            +(zz(kk)-D1_Z)*exp(-(pow(xx(ii)-D1_X,2)+pow(zz(kk)-D1_Z,2))/RAD2) -
             (zz(kk)-D2_Z)*exp(-(pow(xx(ii)-D2_X,2)+pow(zz(kk)-D2_Z,2))/RAD2));
         w = 299.5284/2*(
            -(xx(ii)-D1_X)*exp(-(pow(xx(ii)-D1_X,2)+pow(zz(kk)-D1_Z,2))/RAD2) +
             (xx(ii)-D2_X)*exp(-(pow(xx(ii)-D2_X,2)+pow(zz(kk)-D2_Z,2))/RAD2));
         v = 0;
         /* Add random initial perturbation */
         int myrank;
         MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
         /* Add random noise about 3.5 orders of magnitude below dipole */
         Normal<double> rnd(0,1);
         rnd.seed(myrank);
         for (int i = u.lbound(firstDim); i<= u.ubound(firstDim); i++) {
            for (int j = u.lbound(secondDim); j<= u.ubound(secondDim); j++) {
               for (int k = u.lbound(thirdDim); k<= u.ubound(thirdDim); k++) {
                  u(i,j,k) += 1e-3*rnd.random();
                  v(i,j,k) += 1e-3*rnd.random();
                  w(i,j,k) += 1e-3*rnd.random();
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
      void passive_forcing(double t, DTArray & u, DTArray & u_f, 
            DTArray & v, DTArray & v_f, 
            DTArray & w, DTArray & w_f) {
         u_f = 0;
         v_f = 0;
         w_f = 0;
      }
      userControl(int s):
         szx(s), szy(1), szz(s+1),
         plotnum(0), plot_interval(.005), nextplot(plot_interval),
         lastplot(0),
         itercount(0),last_writeout(0) {
            compute_quadweights(szx,szy,szz,
                  length_x(),length_y(),length_z(),
                  type_x(),type_y(),type_z());
            if (master()) {
               printf("Using array size %d\n",s);
            }
         }
};

int main(int argc, char ** argv) {
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   userControl mycode(atoi(argv[1]));
   FluidEvolve<userControl> ppois(&mycode);
   ppois.initialize();
   ppois.do_run(1.4);
   MPI_Finalize();
   return 0;
}
         
            
