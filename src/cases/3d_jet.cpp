#include "../NSIntegrator.hpp"
#include "../T_util.hpp"
#include "../BaseCase.hpp"
#include <blitz/array.h>
#include <stdio.h>

using namespace NSIntegrator;
using blitz::Array;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

class userControl : public BaseCase {
   public:
   int szx, szy, szz, plotnum;
   double Lx, Ly, Lz, plottime;
   double init_energy;
   bool die_now;
   FILE * logfile;

   /* No tracers */
   int numActive() const { return 0; }
   int numPassive() const { return 0; }
   
   int size_x() const { return szx; }
   int size_y() const { return szy; }
   int size_z() const { return szz; }
   
   double length_x() const { return Lx; }
   double length_y() const { return Ly; }
   double length_z() const { return Lz; }
   double get_visco() const { return 1e-4;}
   double check_timestep(double intime, double now) {
      if (intime < 1e-9 || die_now)
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

      grid = 0*kk + 0*ii + zz(kk);
      write_array(grid,"zgrid");
      write_reader(grid,"zgrid",false);

      fprintf(stderr,"Initializing u\n");
//      u = 1.0/cosh(4*yy(jj))+0*ii+0*kk;
//      u = -0.5*(tanh((yy(jj)-1)/0.1)-tanh((yy(jj)+1)/.1)) + 0*ii+0*kk;
      // 2D planar jet, slightly perturbed in the spanwise direction
      u = 1.0/pow(cosh(yy(jj)/0.2),2) + 0*ii + 0*kk;
      write_array(u,"u_output",plotnum);
      write_reader(u,"u_output",true);
      fprintf(stderr,"Initializing v\n");
      // 3D perturbations -- the v is uneven along z.
      v = .001*(sin(M_PI*xx(ii)/2)  
               +  0.1*sin(M_PI*zz(kk)/2)*sin(M_PI*xx(ii)+0.2)
               + 0.01*sin(M_PI*1.5*xx(ii)-0.4))+0*jj+0*kk;
      write_array(v,"v_output",plotnum);
      write_reader(v,"v_output",true);
      fprintf(stderr,"Initializing w\n");
      w = 0*ii + 0*jj +0*kk;
      write_array(w,"w_output",plotnum);
      write_reader(w,"w_output",true);
      init_energy = sum(u*u+v*v+w*w);
      return;
   }
   
   void vel_analysis(double time, DTArray & u, DTArray & v, DTArray & w) {
      if (time > (plottime - 1e-8*fabs(plottime))) {
         plottime = plottime + 0.2;
         fprintf(stderr,"*");
         plotnum++;
         write_array(u,"u_output",plotnum);
         write_array(v,"v_output",plotnum);
         write_array(w,"w_output",plotnum);
         fflush(logfile);
      }

      double ke = sum(u*u + v*v + w*w);
      double kex = sum(u*u);
      double keh = kex + sum(v*v);
      fprintf(stderr,"%f: (%.2f, %.2g, %.2g) -- (%.2f, %.2f, %.2g) [%g]\n",time,
            max(abs(u)), max(abs(v)), max(abs(w)), kex/ke, keh/ke, 1-keh/ke,
            ke/init_energy);
      fprintf(logfile,"%f: (%.2f, %.2g, %.2g) -- (%.2f, %.2f, %.2g) [%g]\n",
            time, max(abs(u)), max(abs(v)), max(abs(w)), kex/ke, keh/ke, 
            1-keh/ke, ke/init_energy);
      if (ke/init_energy > 1) {
         die_now = true;
      }

      return;
   }

   userControl(): szx(128), szy(128), szz(128), 
                  Lx(4), Ly(8), Lz(4), 
                  plottime(0.2), plotnum(0) {
         die_now = false;
         logfile = fopen("output.log","w");
   };
   ~userControl() {
      fclose(logfile);
   }
};

int main() {
   userControl mycode;
   FluidEvolve<userControl> fluidstuff(&mycode);
   fluidstuff.initialize();
   fluidstuff.do_run(40.0);
   return 0;
}

