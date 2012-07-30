#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include "../BaseCase.hpp"
#include <blitz/array.h>
#include <stdio.h>
#include <mpi.h>

using namespace NSIntegrator;
using blitz::Array;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

class userControl : public BaseCase {
   public:
   int szx, szy, szz, plotnum;
   double Lx, Ly, Lz, plottime;

   DIMTYPE type_y() const { return FREE_SLIP; }

   int size_x() const { return szx; }
   int size_y() const { return szy; }
   int size_z() const { return szz; }
   int numPassive() const { return 1; }
   
   double length_x() const { return Lx; }
   double length_y() const { return Ly; }
   double length_z() const { return Lz; }
   double check_timestep(double intime, double now) {
      if (intime < 1e-9)
         return -1;
      if (intime < 1e-6) {
         fprintf(stderr,"Suspect timestep %f\n",intime);
      }
      double until_plot = plottime - now;
//      intime = fmin(intime,0.001);
      double steps = ceil(until_plot / intime);
      double real_until_plot = steps*intime;
      

      if (fabs(until_plot - real_until_plot) < 1e-8) {
         /* Current timestep will hit plot time in a whole number of
            steps -- or close enough */
         return intime;
      } else {
         /* Adjust timestep */
         return (until_plot/steps);
      }
   }
   void init_tracer(int tnum, DTArray &  tracer) {
      Array <double,1> xx(szx), yy(szy), zz(szz);
      xx = (-Lx/2) + Lx*(ii+0.5)/szx;
      yy = (-Ly/2) + Ly*(ii+0.5)/szy;
      zz = (-Lz/2) + Lz*(ii+0.5)/szz;
      
      tracer = 1.0/pow(cosh((yy(jj)/0.1)),2) + 0*ii + 0*kk;
      write_array(tracer,"t_output",plotnum);
      write_reader(tracer,"t_output",true);
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
//      u = 1.0/cosh(4*yy(jj))+0*ii+0*kk;
//      u = -0.5*(tanh((yy(jj)-1)/0.1)-tanh((yy(jj)+1)/.1)) + 0*ii+0*kk;
      u = 1.0/pow(cosh((yy(jj)/0.2)),2) + 0*ii + 0*kk;
      write_array(u,"u_output",plotnum);
      write_reader(u,"u_output",true);
      fprintf(stderr,"Initializing v\n");
      v = 1e-4*(sin(M_PI*xx(ii)/2)+0.1*sin(M_PI*xx(ii)+0.2)
               +0.01*sin(M_PI*1.5*xx(ii)-0.4))+0*jj+0*kk;
      write_array(v,"v_output",plotnum);
      write_reader(v,"v_output",true);
      fprintf(stderr,"Initializing w\n");
      w = 0;
      return;
   }
   
   void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
         vector<DTArray *> tracer) {
      if (time > (plottime - 1e-8*fabs(plottime))) {
         plottime = plottime + 0.1;
         fprintf(stderr,"*");
         plotnum++;
         write_array(u,"u_output",plotnum);
         write_array(v,"v_output",plotnum);
         write_array(*tracer[0],"t_output",plotnum);
      }


      fprintf(stderr,"%f: (%f, %f)\n",time,
            max(abs(u)), max(abs(v)));

      return;
   }

   userControl(): szx(64), szy(128), szz(1), 
                  Lx(4), Ly(8), Lz(1), 
                  plottime(0.5), plotnum(0) {
   };
};

int main() {
   MPI_Init(0,0);
   userControl mycode;
   FluidEvolve<userControl> fluidstuff(&mycode);
   fluidstuff.initialize();
   fluidstuff.do_run(100.0);
   MPI_Finalize();
   return 0;
}

