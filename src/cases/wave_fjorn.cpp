/* Stability of a wave on picnocline with (initially) stable shear above */

/* This is both a model-check and a novel science run.  IGW shows that,
   in the presence of surface shear, a wave moving counter to that shear can
   cause an instability.  The wave-induced currents cause a vorticity maximum
   in the interior of the domain (when in the absence of a wave the vortcity
   maximum is at the boundary).  This causes the wave to extract energy
   from the background shear.

   Once that is recreated (2d), it will be worth exploring how (and if)
   these instabilities go 3D. */

#include "../TArray.hpp"
#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include "../Science.hpp"
#include "../Par_util.hpp"
#include "../BaseCase.hpp"
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

#define SIZE_X 3072
#define SIZE_Y 1
#define SIZE_Z 512
#define L_X 50
#define L_Y 1
#define L_Z 1

#define PIC_HEIGHT -0.25*L_Z
#define PIC_WIDTH 0.05*L_Z
#define PIC_STRENGTH 0.01

#define BUMP_WIDTH 1.5
#define BUMP_HEIGHT 0.4*L_Z

#define SHEAR_WIDTH 0.05*L_Z
#define SHEAR_STRENGTH 0.17

class userControl : public BaseCase {
   public:
      /* Base parameters: sizes, lengths, plot data */
      int szx, szy, szz, 
          plotnum, itercount;
      double Lx, Ly, Lz, 
             plot_interval, nextplot;

      /* Variables to control stratification */
      double pic_height, pic_strength, pic_width,
             bump_width, bump_amplitude;

      /* Variables to control shear (tanh centered at top) */
      double shear_strength, shear_width;

      /* gravity */
      double g;

      /* Grid arrays*/
      Array<double,1> zz, yy, xx;

      /* Return appropriate sizes to the Navier-Stokes integrator */
      int size_x() const { return szx; }
      int size_y() const { return 1; }
      int size_z() const { return szz; }

      /* Specify dimension types */
      DIMTYPE type_x() const { return PERIODIC; }
      DIMTYPE type_default() const { return FREE_SLIP; }

      /* One active tracer -- density */
      int numActive() const { return 1; }

      /* Domain lengths */
      double length_x() const { return Lx; }
      double length_y() const { return Ly; }
      double length_z() const { return Lz; }

      /* Function to check that the suggested timestep is appropriate. 
         Usually, there's no major changes here -- the most frequent
         is to set the current timestep to evenly divide the plot intervals,
         so that we write data out at sensible times. */
      double check_timestep(double intime, double now) {
         if (intime < 1e-9) {
            /* Something's gone wrong, so abort */
            fprintf(stderr,"Tiny timestep returned, aborting\n");
            return -1;
         } else if (intime > .1) {
            /* Cap the maximum timestep size */
            intime = .1; 
         }
         /* Calculate how many timesteps we'll take until we pass the
            next plottime. */
         double until_plot = nextplot - now;
         double steps = ceil(until_plot / intime); 
         double real_until_plot = steps*intime;

         if (fabs(until_plot - real_until_plot) < 1e-5*plot_interval) {
            /* We'll hit close enough to the plot point, so good enough */
            return intime;
         } else {
            /* Adjust */
            return (until_plot / steps);
         }
      }
      /* Data analysis and output routine */
      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer) {
         /* Write out velocities and density if we've passed the designated
            plot time */
         bool plotted = false; // Check if we plotted this timestep
         if (time > (nextplot - 1e-5*fabs(nextplot))) {
           blitz::Array<double,3> diag(split_range(szx),
                  blitz::Range(1,1),blitz::Range(1,1));
            nextplot = nextplot + plot_interval;
            plotnum++;
            /* Write out velocities and density */
            write_array(u,"u_output",plotnum);
            write_array(v,"v_output",plotnum);
            write_array(w,"w_output",plotnum);
            write_array(*(tracer[0]),"rho_output",plotnum);
            plotted = true;
         }
         itercount++;
         if (!(itercount % 25) || plotted) {
            /* Print out some diagnostic information */
            double mu = psmax(max(abs(u))), // maximum u velocity
                   mv = psmax(max(abs(v))), // maximum v velocity
                   mw = psmax(max(abs(w))), // maximum w velocity 
                   mt = psmax(max(abs(*tracer[0]))), // maximum tracer (density)
                   ke = pssum(sum(u*u+v*v+w*w))/(szx*szy*szz)*Lx*Ly*Lz; // KE
            if (master()) // write the diagnostic to standard out
               fprintf(stdout,"%f [%d]: (%.2g, %.2g, %.2g: %.3f), (%.2g)\n",
                     time, itercount, mu, mv, mw, mt, ke);
         }
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         /* Initialize the velocities */
         /* For initial shear, v and w are 0 to begin with */
         v=0; w=0;
         /* And u is nonzero.  It is a hyperbolic tangent, centered at
            the surface, decaying to 0 in the deep */
//         u = shear_strength*(1 + tanh(zz(kk)/(shear_width))) + 0*ii + 0*jj;
         /* To be completely consistent with BCs, u_z should be 0 at the top.
            So, replace the (1+tanh) with a sech.  This doesn't exactly satisfy
            the "vorticity maximum at boundary" condition, but the instability
            is supposed to be slow */
         u = shear_strength/cosh(zz(kk)/shear_width) + 0*ii + 0*jj;
         write_reader(u,"u_output",true);
         write_reader(v,"v_output",true);
         write_reader(w,"w_output",true);
      }

      void init_tracer(int t_num, DTArray & rho) {
         /* Initialize each tracer.  In this case, we only have one tracer,
            so t_num should be 0.  If we had more (say, heat and salt),
            this function would be called once per tracer */
         assert(t_num == 0);
         /* Initialize a temporary array for the interface displacement */
         Array<double,1> interface(split_range(szx));
         /* And a temporary grid to write out x, y, z coordinates */
         Array<double,3> grid(alloc_lbound(szx,1,szz),
                              alloc_extent(szx,1,szz),
                              alloc_storage(szx,1,szz));
         grid = xx(ii) + 0*jj + 0*kk;
         write_array(grid,"xgrid"); write_reader(grid,"xgrid",false);
         grid = 0*ii + yy(jj) + 0*kk;
         write_array(grid,"ygrid"); write_reader(grid,"ygrid",false);
         grid = 0*ii + 0*jj + zz(kk);
         write_array(grid,"zgrid"); write_reader(grid,"zgrid",false);
         
         /* Calculate the interface perturbation */
         interface = -bump_amplitude / pow(cosh(xx(ii)/bump_width),2);

         /* Initialize the density (rho) */
         rho = (1 - pic_strength * tanh((zz(kk) - pic_height - interface(ii))/pic_width));

         /* Write out the initial density field */
         write_array(rho,"rho_output",0); write_reader(rho,"rho_output",true); 
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

      /* Constructor: initialize several of the physical parameters here */
      /* For now, these variables are initialized to the ones #defined above,
         save for a couple (like gravity) that are hard-coded.  There is no reason,
         however, that these variables cannot be read in from a configuration file
         (or the command line) at program run-time.  This would indeed be a Really
         Nice Way of doing things.  However, it's also more work. */
      userControl() : 
         /* Plot parameters */
         plotnum(0), // initial plot number
         plot_interval(0.1), // time between plots
         nextplot(plot_interval), // time of next plot
         itercount(0), // initiaal iteration count */
         /* Physical parameters */
         g(9.81), // gravity
         Lx(L_X), Ly(L_Y), Lz(L_Z), // box lengths
         szx(SIZE_X), szy(SIZE_Y), szz(SIZE_Z), // grid resolution
         /* Problem prarameters */
         pic_height(PIC_HEIGHT), pic_strength(PIC_STRENGTH), pic_width(PIC_WIDTH),
         bump_amplitude(BUMP_HEIGHT), bump_width(BUMP_WIDTH),
         shear_strength(SHEAR_STRENGTH), shear_width(SHEAR_WIDTH),
         /* Grid arrays */
         zz(szz), yy(szy), xx(split_range(szx)) {
            /* Create grid for use in velocity, tracer initialization */
            xx = -Lx/2 + Lx*(ii+0.5)/szx;
            yy = -Ly/2 + Ly*(ii+0.5)/szy;
            zz = -Lz + Lz*(ii+0.5)/szz;
         }
};

int main() {
   MPI_Init(0,0); // Initialize MPI
   userControl mycode; // Create the userControl object
   EasyFlow fjorn(&mycode); // Create the Navier-Stokes Integrator, using our current code
   fjorn.initialize(); // initialize the integrator
   fjorn.do_run(120); // run to a final time (20)
   MPI_Finalize(); // finalize MPI
   return 0; // return
}
