/* wave_reader_extend.cpp -- as wave_reader, only built to read in 
   2D grids WRITTEN OUT BY SPINS for extension to three dimensions
   with a random perturbation */

#include "../Science.hpp"
#include "../TArray.hpp"
#include "../Par_util.hpp"
#include "../NSIntegrator.hpp"
#include "../BaseCase.hpp"
#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <random/uniform.h>
#include <random/normal.h>


int          Nx, Ny, Nz; // Number of points in x, y, z
double       Lx, Ly, Lz; // Grid lengths of x, y, z 


// Input file names
char xgrid_filename[100],
     ygrid_filename[100],
     zgrid_filename[100],
     u_filename[100],
     v_filename[100],
     w_filename[100],
     rho_filename[100];

// Physical parameters
double g, rot_f, vel_mu, dens_kappa;

// Writeout parameters
double final_time, plot_interval;

// Mapped grid?
bool mapped, threedeevel;

// Perturbation for 3D
double perturbation=0;

// Grid types
DIMTYPE intype_x, intype_y, intype_z;

using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

using ranlib::Normal;

int myrank = -1;

bool restarting = false;
int restart_sequence = 0;
double restart_time = 0;

/* Log arrays, to store some diagnostic information on a per-timestep
   resolution */
//double times_log[100], ke_2d_log[100], ke_3d_log[100], max_v_log[100],
//       max_drho_log[100], pe_log[100];


class userControl : public BaseCase {
   public:
      int plotnum, itercount, lastplot, last_writeout;
      bool plot_now;
      double nextplot;
      double start_rho_spread;

      int size_x() const { return Nx; }
      int size_y() const { return Ny; }
      int size_z() const { return Nz; }

      DIMTYPE type_x() const { return intype_x; }
      DIMTYPE type_y() const { return intype_y; }
      DIMTYPE type_default() const { return intype_z; }

      double length_x() const { return Lx; }
      double length_y() const { return Ly; }
      double length_z() const { return Lz; }

      double get_visco() const {
         return vel_mu;
      }
      double get_diffusivity(int t) const {
         return dens_kappa; 
      }

      double init_time() const { 
         return restart_time;
      }

      bool is_mapped() const {return mapped;}
      void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
         if (master()) 
            fprintf(stderr,"Reading xgrid (%d x %d) from %s\n",Nx,Nz,xgrid_filename);
         read_2d_slice(xg,xgrid_filename,Nx,Nz);
         if (Ny == 1) yg = 0;
         else {
            yg = 0*ii + (0.5+jj)/Ny*Ly-Ly/2 + 0*kk;
         }
         read_2d_slice(zg,zgrid_filename,Nx,Nz);
         if (master())
            fprintf(stderr,"Reading zgrid (%d x %d) from %s\n",Nx,Nz,zgrid_filename);

         // Write out grid readers
         write_array(xg,"xgrid");write_reader(xg,"xgrid",false);
         if (Ny > 1)
            write_array(yg,"ygrid"); write_reader(yg,"ygrid",false);
         write_array(zg,"zgrid"); write_reader(zg,"zgrid",false);
      }

      /* We have an active tracer, namely density */
      int numActive() const { return 1; }

      /* Timestep-check function.  This (long with "write everything" outputs) should
         really be bumped into the BaseCase */
      double check_timestep(double intime, double now) {
//         if (master()) fprintf(stderr,"Input time %g\n",intime);
         if (intime < 1e-9) {
            /* Timestep's too small, somehow stuff is blowing up */
            if (master()) fprintf(stderr,"Tiny timestep (%e), aborting\n",intime);
            return -1;
         } else if (intime > .1) {
            /* Cap the maximum timestep size */
            intime = .1;
         }
         /* Calculate how long we have until the next plottime, and then adjust
            the timestep such that we take a whole number of steps to ge there */
         double until_plot = nextplot - now;
         double steps = ceil(until_plot / intime);
         double real_until_plot = steps*intime;

         if (fabs(until_plot - real_until_plot) < 1e-5*plot_interval) {
            /* Close enough for scientific work */
            return intime;
         } else {
            /* Adjust the timestep */
            return (until_plot / steps);
         }
      }
      void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer, DTArray & pressure) {
         itercount = itercount + 1;
         double mddiff, max_v;

         mddiff = pvmax(*tracer[0]) - pvmin(*tracer[0]);
         max_v = psmax(max(abs(v)));


         if ((time - nextplot) > -1e-5*plot_interval) {
            plot_now = true;
            plotnum = plotnum + 1;
            write_array(u,"u",plotnum);
            if (Ny > 1 || rot_f != 0)
               write_array(v,"v",plotnum);
            write_array(w,"w",plotnum);
            write_array(*tracer[0],"rho",plotnum);
            write_array(pressure,"p",plotnum);
            lastplot = itercount;
            if (master()) {
               FILE * plottimes = fopen("plot_times.txt","a");
               assert(plottimes);
               fprintf(plottimes,"%.10g\n",time);
               fclose(plottimes);
               fprintf(stdout,"*");
            }
            nextplot = nextplot + plot_interval;
         } else {
            plot_now = false;
         }
         if (((itercount - lastplot) % 1 == 0) || plot_now) {
            double mu = psmax(max(abs(u))),
                   mw = psmax(max(abs(w)));
            if (master())
               fprintf(stdout,"%f [%d]: (%.2g, %.2g, %.2g: %.3g)\n",
                     time,itercount,mu,max_v,mw,mddiff);
            last_writeout = itercount;
         }
         //MPI_Finalize(); exit(1);
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         /* Initialize u, w from read-in slice */
         if (master()) fprintf(stderr,"Initializing velocities\n");
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* u */
            snprintf(filename,100,"u.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(u,filename,Nx,Ny,Nz);

            /* v */
            snprintf(filename,100,"v.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading v from %s\n",filename);
            read_array(v,filename,Nx,Ny,Nz);

            /* w */
            snprintf(filename,100,"w.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading w from %s\n",filename);
            read_array(w,filename,Nx,Ny,Nz);
            return;
         }
         if (master()) fprintf(stderr,"Reading u (%d x %d) from %s\n",Nx,Nz,u_filename);
         read_2d_restart(u,u_filename,Nx,Nz);
         if (master()) fprintf(stderr,"Reading w (%d x %d) from %s\n",Nx,Nz,w_filename);
         read_2d_restart(w,w_filename,Nx,Nz);
         if (rot_f != 0) {
            if (master()) fprintf(stderr,"Reading v (%d x %d) from %s\n",Nx,Nz,v_filename);
            read_2d_restart(v,v_filename,Nx,Nz);
         } else {
            v = 0;
         }
         // Add a random perturbation to trigger any 3D instabilities
         {
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
            Normal<double> rnd(0,1);
            for (int i = u.lbound(firstDim); i <= u.ubound(firstDim); i++) {
               rnd.seed(i);
               for (int j = u.lbound(secondDim); j <= u.ubound(secondDim); j++) {
                  for (int k = u.lbound(thirdDim); k <= u.ubound(thirdDim); k++) {
                     u(i,j,k) *= 1+perturbation*rnd.random();
                     v(i,j,k) *= 1+perturbation*rnd.random();
                     w(i,j,k) *= 1+perturbation*rnd.random();
                  }
               }
            }
         }
         /* Write out initial values */
         write_array(u,"u",0);
         write_reader(u,"u",true);
         write_array(v,"v",0);
         write_reader(v,"v",true);
         write_array(w,"w",0);
         write_reader(w,"w",true);
      }
      void init_tracer(int t_num, DTArray & rho) {
         if (master()) fprintf(stderr,"Initializing tracer\n");
         /* Initialize the density and take the opportunity to write out the grid */
         assert(t_num == 0);
         // Generate the grid
         if (type_x() == NO_SLIP) {
            // Chebyshev-ordering, 0->Lx
            rho = length_x()*(0.5+0.5*cos(M_PI*ii/(size_x()-1))) + 0*kk;
         } else {
            // Fourier-grid, 0->Lx
            rho = (0.5+ii)*length_x()/size_x();
         }
         write_array(rho,"xgrid");write_reader(rho,"xgrid",false);

         // y-grid
         rho = (0.5+jj)*length_y()/size_y();
         write_array(rho,"ygrid");write_reader(rho,"ygrid",false);

         // z-grid
         if (type_z() == NO_SLIP) {
            rho = -length_z()*(0.5+0.5*cos(M_PI*kk/(size_z()-1)));
         } else {
            rho = (0.5+kk)*length_z()/size_z();
         }
         write_array(rho,"zgrid");write_reader(rho,"zgrid",false);
         /*
         if (!mapped) {
            // First, if not mapped grid, commandeer rho to write out grids
            if (master()) 
               fprintf(stderr,"Reading xgrid (%d x %d) from %s\n",Nx,Nz,xgrid_filename);
            read_2d_slice(rho,xgrid_filename,Nx,Nz);
            // Write out grid readers
            write_array(rho,"xgrid");write_reader(rho,"xgrid",false);
            if (Ny > 1){
               rho = 0*ii + (0.5+jj)/Ny*Ly - Ly/2 + 0*kk;
               write_array(rho,"ygrid"); write_reader(rho,"ygrid",false);
            }
            read_2d_slice(rho,zgrid_filename,Nx,Nz);
            if (master())
               fprintf(stderr,"Reading zgrid (%d x %d) from %s\n",Nx,Nz,zgrid_filename);

            write_array(rho,"zgrid"); write_reader(rho,"zgrid",false);
         } */
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* rho */
            snprintf(filename,100,"rho.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading rho from %s\n",filename);
            read_array(rho,filename,Nx,Ny,Nz);
            return;
         }
         if (master()) fprintf(stderr,"Reading rho (%d x %d) from %s\n",Nx,Nz,rho_filename);
         read_2d_restart(rho,rho_filename,Nx,Nz); 
         write_array(rho,"rho",0);
         write_reader(rho,"rho",true);
         write_reader(rho,"p",true);
      }

      void forcing(double t, DTArray & u, DTArray & u_f,
          DTArray & v, DTArray & v_f, DTArray & w, DTArray & w_f,
          vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
         /* Velocity forcing */
         u_f = -rot_f * v; v_f = +rot_f * u;
         w_f = -g*((*tracers[0]));
         *(tracers_f[0]) = 0;
      }

      userControl() :
         plotnum(restart_sequence), 
         nextplot(plot_interval*(plotnum+1)), itercount(0), lastplot(0),
         last_writeout(0) 
         {
            compute_quadweights(size_x(),size_y(),size_z(),
                  length_x(),length_y(),length_z(),
                  type_x(),type_y(),type_z());
         }
};

int main(int argc, char ** argv) {
   /* Set implicit filtering */
   //f_strength = -.25;
   //f_order = 4;
   //f_cutoff = 0.6;
   MPI_Init(&argc,&argv);
   /* Since we have a few parameters to look at, open a configuration
      file */
   FILE * config_file = fopen("config_file","r");
   assert(config_file);
   /* Configuration file specification is as follows:
      mapped?  % MAPPED / UNMAPPED 
      type_x Nx Lx xgrid_filename
      type_y Ny Ly ygrid_filename
      type_z Nz Lz zgrid_filename
      filename_u
      filename_v
      filename_w
      filename_rho
      g rot_f vel_mu dens_kappa
      final_time
      plot_interval 
      perturbation
    */
   {
      char mapping[100]; int unused;
      // Whether the grid is mapped or unmapped
      unused = fscanf(config_file," %s ",mapping);
      if (!strcmp(mapping,"MAPPED")) {
         mapped = true;
         threedeevel = false;
      } else if (!strcmp(mapping,"UNMAPPED")) {
         mapped = false;
         threedeevel = false;
      } else {
         if (master()) {
            fprintf(stderr,"Error in configuration file: cannot tell whether %s\n",mapping);
            fprintf(stderr,"is MAPPED or UNMAPPED\n");
         }
         MPI_Finalize(); exit(1);
      }
      if (master()) fprintf(stderr,"Received %s grid\n",(mapped?"mapped":"unmapped"));

      // x direction
      char type_str[100];
      unused = fscanf(config_file," %s %d %lf %s ",type_str,&Nx,&Lx,xgrid_filename);
      if (master()) fprintf(stderr,"In x, received a grid of type ");
      // x-type conversion
      if (!strcmp(type_str ,"PERIODIC")) {
         intype_x = PERIODIC;
         if (master()) fprintf(stderr,"Fourier periodic");
      } else if (!strcmp(type_str,"FOURIER_SLIP")) {
         intype_x = FREE_SLIP;
         if (master()) fprintf(stderr,"Fourier slip");
      } else if (!strcmp(type_str,"CHEBY_SLIP") || !strcmp(type_str,"CHEBY_NOSLIP")) {
         intype_x = NO_SLIP;
         if (master()) fprintf(stderr,"Chebyshev");
      } else {
         if (master()) fprintf(stderr,"invalid! (%s)",type_str);
         MPI_Finalize(); exit(1);
      }

      if (master()) fprintf(stderr," with %d points, length %g, filename %s\n",Nx,Lx,xgrid_filename);
      if (Nx <= 0 || Lx == 0) {
         MPI_Finalize(); exit(1);
      }
      // Total hack, see Lz for explanation
      if (intype_x == NO_SLIP) {
         Lx = -Lx;
      }

      // y direction
      unused = fscanf(config_file," %s %d %lf %s ",type_str,&Ny,&Ly,ygrid_filename);
      if (master()) fprintf(stderr,"In y, received a grid of type ");
      // x-type conversion
      if (!strcmp(type_str ,"PERIODIC")) {
         intype_y = PERIODIC;
         if (master()) fprintf(stderr,"Fourier periodic");
      } else if (!strcmp(type_str,"FOURIER_SLIP")) {
         intype_y = FREE_SLIP;
         if (master()) fprintf(stderr,"Fourier slip");
      } else if (!strcmp(type_str,"CHEBY_SLIP") || !strcmp(type_str,"CHEBY_NOSLIP")) {
         intype_y = NO_SLIP;
         if (master()) fprintf(stderr,"Chebyshev");
      } else {
         if (master()) fprintf(stderr,"invalid! (%s)",type_str);
         MPI_Finalize(); exit(1);
      }

      if (master()) fprintf(stderr," with %d points, length %g, filename %s\n",Ny,Ly,ygrid_filename);
      if (Ny <= 0 || Ly == 0) {
         MPI_Finalize(); exit(1);
      }
      // z direction
      unused = fscanf(config_file," %s %d %lf %s ",type_str,&Nz,&Lz,zgrid_filename);
      if (master()) fprintf(stderr,"In z, received a grid of type ");
      // z-type conversion
      if (!strcmp(type_str ,"PERIODIC")) {
         intype_z = PERIODIC;
         if (master()) fprintf(stderr,"Fourier periodic");
      } else if (!strcmp(type_str,"FOURIER_SLIP")) {
         intype_z = FREE_SLIP;
         if (master()) fprintf(stderr,"Fourier slip");
      } else if (!strcmp(type_str,"CHEBY_SLIP") || !strcmp(type_str,"CHEBY_NOSLIP")) {
         intype_z = NO_SLIP;
         if (master()) fprintf(stderr,"Chebyshev");
      } else {
         if (master()) fprintf(stderr,"invalid! (%s)",type_str);
         MPI_Finalize(); exit(1);
      }

      if (master()) fprintf(stderr," with %d points, length %g, filename %s\n",Nz,Lz,zgrid_filename);
      if (Nz <= 0 || Lz == 0) {
         MPI_Finalize(); exit(1);
      }
      // This line reflects an ugly, ugly hack.  If we're using unmapped, Chebyshev grids, then logic
      // suggets that the array origin -- the (1,1) point -- should be at the bottom-left of the domain.
      // However, the traditional Chebyshev ordering has this in the reverse order.  Until now, this has
      // been solved by defining the grid in the physical ordering and just dealing with it in the 
      // underlying code, using a negative "base grid length" during differentiation.  This won't do when
      // we're not generating our own grid -- plain Matlab code will give a top-right origin.  So to
      // take this into account without breaking already-written cases, cover up one hack with another
      // and incorporate a negative grid length.
      if (intype_z == NO_SLIP) {
         Lz = -Lz;
      }

      // Field filenames
      unused = fscanf(config_file,"%s %s %s %s\n",u_filename,v_filename,w_filename,rho_filename);
      if (master()) fprintf(stderr,"Input file names:\nu: %s\nv: %s\nw: %s\nrho: %s\n",u_filename,
            v_filename, w_filename, rho_filename);

      // Physical parameters
      unused = fscanf(config_file,"%lf %lf %lf %lf",&g,&rot_f,&vel_mu,&dens_kappa);
      if (master()) fprintf(stderr,"Physical parameters:\ng: %g\nf: %g\nnu: %g\nkappa: %g\n",
            g,rot_f,vel_mu,dens_kappa);
      if (vel_mu < 0 || dens_kappa < 0) {
         MPI_Finalize(); exit(1);
      }

      unused = fscanf(config_file,"%lf %lf",&final_time,&plot_interval);
      if (master()) fprintf(stderr,"Final time %g, plot interval %f\n",final_time,plot_interval);
      if (final_time <= 0 || plot_interval <= 0) {
         MPI_Finalize(); exit(1);
      }

      unused = fscanf(config_file,"%lf",&perturbation);
      if (master()) fprintf(stderr,"3D perturbation factor of %g\n",perturbation);


   }


   fclose(config_file);
   //MPI_Finalize(); exit(1);
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
   userControl mycode;
   FluidEvolve<userControl> kevin_kh(&mycode);
   kevin_kh.initialize();
   kevin_kh.do_run(final_time);
   MPI_Finalize();
}
