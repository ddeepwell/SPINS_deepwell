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
      /* Grid sizes and plot statistics */
      int szx, szy, szz, plotnum, itercount, lastplot, last_writeout;
      double plot_interval, nextplot;

      // Gradient operator
      Grad * gradient_op;

      // 1D grid arrays
      Array<double,1> xx, yy, zz;

      // Array for vorticity computation
      DTArray vorticity;

      // A no-slip box, with periodic spanwise
      DIMTYPE type_x() const { return NO_SLIP; }
      DIMTYPE type_y() const { return PERIODIC; }
      DIMTYPE type_z() const { return NO_SLIP; }

      // Use a grid-scale Rynolds-number of 1250
      double get_visco() const {
         return 1.0/1250;
      }

      // Give a 2x1x2 box
      double length_x() const { return 2; }
      double length_y() const { return 1; }
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
         // Now, calculate how many timesteps remain until the next writeout
         double until_plot = nextplot - now;
         int steps = ceil(until_plot / intime);
         // And calculate where we will actually be after (steps) timesteps
         // of the current size
         double true_fintime = steps*intime;

         // If that's close enough to the real writeout time, that's fine.
         if (fabs(until_plot-true_fintime) < 1e-6) {
            return intime;
         } else {
            // Otherwise, square up the timeteps.  This will always shrink the timestep.
            return (until_plot / steps);
         }
      }

      // Record the gradient-taking object.  This is given by the NSIntegrator
      // code, and it reflects the boundary types and any Jacobian-transform
      void set_grad(Grad * in_grad) {
         gradient_op = in_grad;
      }

      void compute_vorticity(DTArray & u, DTArray & w) {
         // Compute vorticity 
         gradient_op->setup_array(&u,CHEBY,FOURIER,CHEBY);
         // Put du/dz in vorticity
         gradient_op->get_dz(&vorticity,false);
         // Invert that to get -du/dz
         vorticity = vorticity*(-1);
         // And add dw/dx
         gradient_op->setup_array(&w,CHEBY,FOURIER,CHEBY);
         gradient_op->get_dx(&vorticity,true);
      }


      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *>  tracer, DTArray & pressure) {
         /* Write out velocities */
         bool plotted = false;
         itercount++;

         // Compute the 2D vorticity
         compute_vorticity(u,w);

         double enst, ke, ke2d;
         // Compute enstrophy, sum(vort^2)
         enst = enst_record[itercount-last_writeout-1] = 
            pssum(sum((*get_quad_x())(ii)*(*get_quad_y())(jj)*
                     (*get_quad_z())(kk)*vorticity*vorticity));

         // And KE sum(u^2+w^2).  It needs to be divided by 2 for true energy.
         ke = ke_record[itercount-last_writeout-1] = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)*
                  (pow(u(ii,jj,kk),2)+pow(v(ii,jj,kk),2)+pow(w(ii,jj,kk),2))));

         // KE in the spanwise mean of velocity.  For a 2D run (szy=1), this will
         // be identical to the full KE.
         ke2d = ke2d_record[itercount-last_writeout-1] = pssum(sum(
                  (*get_quad_x())(ii)*(*get_quad_z())(jj)*
                  (pow(mean(u(ii,kk,jj),kk),2)+
                   pow(mean(v(ii,kk,jj),kk),2)+
                   pow(mean(w(ii,kk,jj),kk),2))))*(length_y());

         // The current time.
         times_record[itercount-last_writeout-1] = time;

         
         if ((time - nextplot) > -1e-5*plot_interval) {
            plotted = true;
            nextplot += plot_interval;
            if (master()) fprintf(stdout,"*");
            plotnum++;
            // Write u, v, w, and vorticity to disk
            write_array(u,"u",plotnum);
            write_array(v,"v",plotnum);
            write_array(w,"w",plotnum);
            write_array(vorticity,"vort",plotnum);
            lastplot = itercount;
            if (master()) {
               // And save the current timestep for reference
               FILE * plottimes = fopen("plot_times.txt","a");
               assert(plottimes);
               fprintf(plottimes,"%.10g\n",time);
               fclose(plottimes);
            }
         }
         if ((itercount - lastplot)%1 == 0 || plotted) {
            double mu = psmax(max(abs(u))),
                   mv = psmax(max(abs(v))),
                   mw = psmax(max(abs(w)));
            // Diagnostic information -- write out maximum (absolute) velocities
            // along with enstrophy and KE on a per-timestep basis.  This allows
            // for very finely-detailed comparisons between runs.
            if (master())
               fprintf(stdout,"%f [%d] (%.4g, %.4g, %.4g) -- (%g, %g, %g)\n",
                     time, itercount, mu, mv, mw,enst,ke,ke2d);
            if (master()) {
               FILE * en_record = fopen("energy_record.txt","a");
               assert(en_record);
               for (int i = 0; i < (itercount-last_writeout); i++) {
                  fprintf(en_record,"%.9g %.9g %.9g %.9g\n",times_record[i],
                        ke_record[i],enst_record[i],ke2d_record[i]);
               }
               fclose(en_record);
            }
            last_writeout = itercount; 
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         xx = -length_z()/2*cos(M_PI*ii/(szx-1));
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
         /* Add random initial perturbation, if this is a 3D run */
         /* This noise is neither incompressible nor satisfying of the boundary
            conditions.  The code will make it do both after the first timestep.*/
         if (szy > 1) {
            // Why add noise only in 3D?  First, comparison with Clercx and Bruneu
            // [2006] uses noise-free runs.  Secondly, the instability triggered
            // here is three-dimensional.
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
         }

         // Write out the initial arrays: velocity...
         write_array(u,"u",0);
         write_reader(u,"u",true);
         write_array(v,"v",0);
         write_reader(v,"v",true);
         write_array(w,"w",0);
         write_reader(w,"w",true);
         // ... and vorticity
         compute_vorticity(u,w);
         write_array(vorticity,"vort",0);
         write_reader(vorticity,"vort",true);
         grid = xx(ii) + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = yy(jj) + 0*kk;
         write_array(grid,"ygrid"); write_reader(grid,"ygrid",false);
         grid = zz(kk);
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);
      }
      // Once initialized, this is freely-evolving flow.  No forcing is necessary
      void passive_forcing(double t, DTArray & u, DTArray & u_f, 
            DTArray & v, DTArray & v_f, 
            DTArray & w, DTArray & w_f) {
         u_f = 0;
         v_f = 0;
         w_f = 0;
      }
      userControl(int s):
         // Setup a 2D run, of size S x 1 x S
         szx(s), szy(1), szz(s),
         // Write out fields every 0.005 timeunits for detailed graphics
         plotnum(0), plot_interval(.005), 
         nextplot(plot_interval), lastplot(0),
         itercount(0),last_writeout(0),
         // Initalize arrays for 1D grid coordinates
         xx(split_range(szx)), yy(szy), zz(szz),
         // Initalize the 2(3)D array for vorticity computation
         vorticity(alloc_lbound(szx,szy,szz),
               alloc_extent(szx,szy,szz),
               alloc_storage(szx,szy,szz)) {
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
   f_strength = -.25;
   f_order = 4;
   f_cutoff = 0.8;
   // Read in the grid size from the command-line
   int grid_size = 0;
   if (argc > 1) grid_size = atoi(argv[1]);
   // And use a sensible default if it's not given or invalid
   if (grid_size <= 0) grid_size = 64;
   userControl mycode(grid_size);
   FluidEvolve<userControl> ppois(&mycode);
   ppois.initialize();
   ppois.do_run(1.4);
   MPI_Finalize();
   return 0;
}
         
            
