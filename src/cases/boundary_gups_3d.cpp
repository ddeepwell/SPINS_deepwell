/* Interaction of a solitary wave with a no-slip boundary, given background
   shear relaxing from a uniform current */

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

#define G 9.81

#define L_X 10
#define L_Y 0.4
#define L_Z 1

#define NOISE_MAG 1e-4

double PHASE_SPEED = 0;


int myrank = -1;

bool restarting = false;
int restart_sequence = 0;
double restart_time = 0;

double times_record[100];

class userControl : public BaseCase {
   public:
      int szx, szy, szz, plotnum, itercount, lastplot, last_writeout;
      double plot_interval, nextplot;

      Array<double,1> xx,yy,zz;

      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_y() const { return PERIODIC; }
      DIMTYPE type_z() const { return NO_SLIP; }

      double get_visco() const {
         return 1e-5;
      }
      double get_diffusivity(int t) const {
         return 1e-5/7; // 1e-5 / 7, as a thermocline
      }
      double length_x() const { return L_X; }
      double length_y() const { return L_Y; }
      double length_z() const { return L_Z; }

      int size_x() const { return szx; }
      int size_y() const { return szy; }
      int size_z() const { return szz; }

      double init_time() const {
         if (!restarting) return 0;
         else return restart_time;
      }

      int numActive() const { return 1; }

      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            if (master()) fprintf(stderr,"Tiny timestep, aborting\n");
            return -1;
         } else if (itercount < 10 && intime > .01) {
            intime = 1e-3;
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
//         DTArray * vor_x, * vor_y, *vor_z;
//         vorticity(u,v,w,vor_x,vor_y,vor_z,length_x(),length_y(),length_z(),
//            size_x(),size_y(),size_z(),type_x(),type_y(),type_z());
//
//         double enst, ke, ke2d;
//         enst = enst_record[itercount-last_writeout-1] = pssum(sum(
//                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)*
//                  (pow((*vor_x)(ii,jj,kk),2) + pow((*vor_y)(ii,jj,kk),2) +
//                   pow((*vor_z)(ii,jj,kk),2))));
//         ke = ke_record[itercount-last_writeout-1] = pssum(sum(
//                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)*
//                  (pow(u(ii,jj,kk),2)+pow(v(ii,jj,kk),2)+pow(w(ii,jj,kk),2))));
//         ke2d = ke2d_record[itercount-last_writeout-1] = pssum(sum(
//                  (*get_quad_x())(ii)*(*get_quad_z())(jj)*
//                  (pow(mean(u(ii,kk,jj),kk),2)+
//                   pow(mean(v(ii,kk,jj),kk),2)+
//                   pow(mean(w(ii,kk,jj),kk),2))))*(length_y());

         times_record[itercount-last_writeout-1] = time;

         
         if ((time - nextplot) > -1e-5*plot_interval) {
            plotted = true;
            nextplot += plot_interval;
            if (master()) fprintf(stdout,"*");
            plotnum++;
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(w,"w_output",plotnum);
            write_array(*tracer[0],"rho_output",plotnum);
            write_array(pressure,"p_output",plotnum);
            lastplot = itercount;
            if (master()) {
               FILE * plottimes = fopen("plot_times.txt","a");
               assert(plottimes);
               fprintf(plottimes,"%.10g\n",time);
               fclose(plottimes);
            }
         }
         if ((itercount - lastplot)%5 == 0 || plotted) {
            double mu = psmax(max(abs(u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w))),
                   mt = pvmax(*tracer[0]) - pvmin(*tracer[0]);
            if (master())
               fprintf(stdout,"%f [%d] (%.4g, %.4g, %.4g, %.4g)\n",
                     time, itercount, mu, mv, mw, mt);;
            if (master()) {
               FILE * en_record = fopen("timesteps_record.txt","a");
               assert(en_record);
               for (int i = 0; i < (itercount-last_writeout); i++) {
                  fprintf(en_record,"%g\n",times_record[i]);
               }
               fclose(en_record);
            }
            last_writeout = itercount;
         }
      }

      void init_tracer(int t_num, DTArray & rho) {
         assert(t_num == 0);
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* rho */
            snprintf(filename,100,"rho_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(rho,filename,szx,szy,szz);
            return;
         }

         Array<double,3> read_vels(split_range(szx),blitz::Range(0,0),blitz::Range(0,size_z()-1));
         if (master()) fprintf(stdout,"reading input rho\n");
         read_array(read_vels,"input_rho",szx,1,szz);
         for (int j = rho.lbound(secondDim); j <= rho.ubound(secondDim); j++) {
            rho(blitz::Range::all(),blitz::Range(j,j),blitz::Range::all()) =
               read_vels(blitz::Range::all(),blitz::Range(0,0),
                     blitz::Range::all());
         }
         
         write_array(rho,"rho_output",0); write_reader(rho,"rho_output",true);
         write_reader(rho,"p_output",true);
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* u */
            snprintf(filename,100,"u_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(u,filename,szx,szy,szz);

            /* v */
            snprintf(filename,100,"v_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading v from %s\n",filename);
            read_array(v,filename,szx,szy,szz);

            /* w */
            snprintf(filename,100,"w_output.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading w from %s\n",filename);
            read_array(w,filename,szx,szy,szz);
            return;
         } 

         Array<double,3> read_vels(split_range(szx),blitz::Range(0,0),blitz::Range(0,size_z()-1));
         if (master()) fprintf(stdout,"reading input u\n");
         read_array(read_vels,"input_u",szx,1,szz);
         for (int j = u.lbound(secondDim); j <= u.ubound(secondDim); j++) {
            u(blitz::Range::all(),blitz::Range(j,j),blitz::Range::all()) =
               read_vels(blitz::Range::all(),blitz::Range(0,0),
                     blitz::Range::all());
         }
         v = 0;
         if (master()) fprintf(stdout,"Reading input w\n");
         read_array(read_vels,"input_w",szx,1,szz);
         for (int j = w.lbound(secondDim); j <= w.ubound(secondDim); j++) {
            w(blitz::Range::all(),blitz::Range(j,j),blitz::Range::all()) =
               read_vels(blitz::Range::all(),blitz::Range(0,0),
                     blitz::Range::all());
         }
         /* Add random initial perturbation */
         int myrank;
         MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
         /* Add random noise about 3 orders of magnitude below dipole */
         Normal<double> rnd(0,1);
         rnd.seed(myrank);
         for (int i = u.lbound(firstDim); i<= u.ubound(firstDim); i++) {
            for (int j = u.lbound(secondDim); j<= u.ubound(secondDim); j++) {
               for (int k = u.lbound(thirdDim); k<= u.ubound(thirdDim); k++) {
                  u(i,j,k) += NOISE_MAG*rnd.random()-PHASE_SPEED;
                  v(i,j,k) = NOISE_MAG*rnd.random();
                  w(i,j,k) += NOISE_MAG*rnd.random();
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
         Array<double,3> grid(alloc_lbound(szx,szy,szz),
               alloc_extent(szx,szy,szz),
               alloc_storage(szx,szy,szz));
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
         w_f = -G*(*tracers[0]);
      }
      void tracer_forcing(double t, const DTArray & u, const DTArray & v,
            const DTArray & w, vector<DTArray *> & tracers_f) {
         *tracers_f[0] = 0;
      }
      userControl(int s1, int s2, int s3):
         szx(s1), szy(s2), szz(s3),
         plotnum(0), itercount(0), lastplot(0), last_writeout(0),
         plot_interval(.25), nextplot(plot_interval),
         xx(split_range(szx)), yy(szy), zz(szz) {
            xx = -length_x()/2 + length_x()*(ii+0.5)/size_x();
            yy = -length_y()/2 + length_y()*(ii+0.5)/size_y();
            zz = length_z()/2-length_z()/2*cos(M_PI*ii/(size_z()-1));
            compute_quadweights(szx,szy,szz,
                  length_x(),length_y(),length_z(),
                  type_x(),type_y(),type_z());
            if (master()) {
               printf("Using array size %dx%dx%d\n",szx,szy,szz);
            }
            if (restarting) {
               nextplot = restart_time + plot_interval;
               plotnum = restart_sequence;
            }
         }
};

int main(int argc, char ** argv) {
   MPI_Init(&argc, &argv);
   /* Set implicit filtering */
   f_strength = -.25;
   f_order = 4;
   f_cutoff = 0.8;
   if (argc > 3) {
      /* Possibly restarting */
      if (strcmp("--restart",argv[1])==0) {
         if (master()) fprintf(stdout,"Restarting...\n");
         restarting = true;
         restart_time = atof(argv[2]);
         if (master()) fprintf(stdout,"Restart time %f\n",restart_time);
         restart_sequence = atoi(argv[3]);
         if (master()) fprintf(stdout,"Restart sequence %d\n",restart_sequence);
      }
   }
   userControl mycode(2048,96,384);
//   userControl mycode(16,1);
   FluidEvolve<BaseCase> ppois(&mycode);
   ppois.initialize();
   ppois.do_run(40);
   MPI_Finalize();
   return 0;
}
         
            
