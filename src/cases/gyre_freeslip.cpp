#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include "../BaseCase.hpp"
#include <blitz/array.h>
#include <stdio.h>
#include <vector>

using namespace NSIntegrator;
using blitz::Array;
using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

class userControl : public BaseCase{
   public:
   int szx, szy, szz, plotnum, sign;
   bool initialized;
   double Lx, Ly, Lz, plottime, plot_interval;
   double mode_x, mode_y;
   double t_now;
   vector<Array<double,3> *> stored_tracers;

   static const int used_tracers=3;

   int size_x() const { return szx; }
   int size_y() const { return szy; }
   int size_z() const { return szz; }
   
   DIMTYPE type_x() const { return FREE_SLIP ;}
   DIMTYPE type_y() const { return FREE_SLIP ;}
//   DIMTYPE type_z() const { return PERIODIC ;}
   DIMTYPE type_default() const { return FREE_SLIP ; }
   
   int numPassive() const { return used_tracers; }
   
   double length_x() const { return Lx; }
   double length_y() const { return Ly; }
   double length_z() const { return Lz; }
   
   double check_timestep(double intime, double now) {
      if (intime < 1e-9)
         return -1;
      if (intime < 1e-6) {
         fprintf(stderr,"Suspect timestep %.2e\n",intime);
      }
      double until_plot = plottime - now;
//      intime = fmin(intime,0.001);
      double steps = ceil(until_plot / intime);
      double real_until_plot = steps*intime;
      
//      fprintf(stderr,"%d",int(steps));

      if (fabs(until_plot - real_until_plot) < 1e-8) {
         /* Current timestep will hit plot time in a whole number of
            steps -- or close enough */
//         fprintf(stderr,"\n");
         return intime;
      } else {
         /* Adjust timestep */
//         fprintf(stderr,"*\n");
         return (until_plot/steps);
      }
   }
   void init_tracer(int tnum, DTArray & tracer) {
      Array <double,1> xx(szx), yy(szy), zz(szz);
      xx = (-Lx/2) + Lx*(ii+0.5)/szx;
      yy = (-Ly/2) + Ly*(ii+0.5)/szy;
      zz = (-Lz/2) + Lz*(ii+0.5)/szz;
      if (initialized) {
        tracer = *(stored_tracers[tnum]);
      } else {
         initialized = true;
         switch (tnum) {
            case 0:
               tracer = exp(-64*(xx(ii)*xx(ii) + yy(jj)*yy(jj))) + 0*kk;
               write_array(tracer,"t0_output",plotnum);
               write_reader(tracer,"t0_output",true);
               break;
            case 1:
               tracer = exp(-64*(xx(ii)*xx(ii) +
                                 pow(yy(jj)-0.2,2))) + 0*kk;
               write_array(tracer,"t1_output",plotnum);
               write_reader(tracer,"t1_output",true);
               break;
            case 2:
               tracer = exp(-64*(xx(ii)*xx(ii) +
                                 pow(yy(jj)+0.2,2))) + 0*kk;
               write_array(tracer,"t2_output",plotnum);
               write_reader(tracer,"t2_output",true);
               break;
            default:
               abort();
         }
      }
   }
   

   void init_vels(DTArray & u, DTArray & v, DTArray & w) {
      Array <double,1> xx(szx), yy(szy), zz(szz);
      xx = (-Lx/2) + Lx*(ii+0.5)/szx;
      yy = (-Ly/2) + Ly*(ii+0.5)/szy;
      zz = (-Lz/2) + Lz*(ii+0.5)/szz;

      Array<double,3> grid(szx,szy,szz);
      grid = xx(ii) + 0*jj + 0*kk;
      write_array(grid,"xgrid");
      write_reader(grid,"xgrid",false);

      grid = yy(jj) + 0*ii + 0*kk;
      write_array(grid,"ygrid");
      write_reader(grid,"ygrid",false);

      grid = 0*kk + 0*ii + 0*jj;

      fprintf(stderr,"Initializing u\n");
      double wavenum_x, wavenum_y, norm;
      wavenum_x = (2*M_PI)*mode_x/Lx;
      wavenum_y = (2*M_PI)*mode_y/Ly;
      norm = fmin(1/wavenum_x, 1/wavenum_y);
      fprintf(stderr,"x: %.2f, y: %.2f, normalized: %.2f\n",
            wavenum_x, wavenum_y, norm);
      // For reference, streamfunction psi is of cos(x)*sin(y) form
      u = sign*norm*wavenum_y*cos(wavenum_x*xx(ii))*sin(wavenum_y*yy(jj)) + 0*kk;
      fprintf(stderr,"max_u: %.2f\n",max(u));
      Array<double,1> tmp(szy);
      tmp = u(0,blitz::Range::all(),0);
//      std::cout << tmp << std::endl;
//      fprintf(stderr,"u(%f, %f): %f\n",xx(szx/2),yy(szy/4),u(szx/2,szy/4,0));
      fprintf(stderr,"Initializing v\n");
      v = -sign*norm*wavenum_x*sin(wavenum_x*xx(ii))*cos(wavenum_y*yy(jj)) + 0*kk;
      if (!initialized) {
         write_array(u,"u_output",plotnum);
         write_reader(u,"u_output",true);
         write_array(v,"v_output",plotnum);
         write_reader(v,"v_output",true);
      }
      fprintf(stderr,"max_v: %.2f\n",max(v));
      fprintf(stderr,"Initializing w\n");
      w = 0;
      return;
   }
   void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
         vector<DTArray *> tracer) {
      if (time > (plottime - 1e-8*fabs(plottime))) {
         t_now = time;
         plottime = plottime + plot_interval;
         fprintf(stderr,"*");
         plotnum++;
         write_array(u,"u_output",plotnum);
         write_array(v,"v_output",plotnum);
         if (used_tracers > 0)  {
            write_array(*tracer[0],"t0_output",plotnum);
            *stored_tracers[0] = *tracer[0];
         }
         if (used_tracers > 1)  {
            write_array(*tracer[1],"t1_output",plotnum);
            *stored_tracers[1] = *tracer[1];
         }
         if (used_tracers > 2)  {
            write_array(*tracer[2],"t2_output",plotnum);
            *stored_tracers[2] = *tracer[2];
         }
      fprintf(stderr,"%f: (%.2f, %.2f) - (%.2f)\n",time,
            max(abs(u)), max(abs(v)), max(abs(*tracer[0])));
      }



      return;
   }

   userControl(): szx(64), szy(68), szz(1), 
                  Lx(2), Ly(2), Lz(1), 
                  plottime(0.025), plot_interval(0.025), plotnum(0),
                  mode_x(.5), mode_y(.5),
                  sign(1), initialized(false), 
                  t_now(0),
                  stored_tracers(used_tracers) {
      for (int ii = 0; ii < used_tracers; ii++) {
         stored_tracers[ii] = new Array<double,3>(szx,szy,szz);
      }
   };

   ~userControl() {
      for (int ii = 0; ii < stored_tracers.size(); ii++) {
         delete stored_tracers[ii];
      }
   }

   void flip() {
      sign = -sign;
   }
};

int main() {
   userControl mycode;
   FluidEvolve<userControl> forward_gyre(&mycode);
   forward_gyre.initialize();
   forward_gyre.do_run(5.0);
/*   mycode.flip();
   FluidEvolve<userControl> backward_gyre(&mycode);
   backward_gyre.initialize();
   backward_gyre.do_run(10.0);*/
   return 0;
}

