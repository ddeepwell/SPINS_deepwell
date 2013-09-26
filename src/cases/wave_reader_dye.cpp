/* wave_reader.cpp -- general case for looking at the evolution of
   waves, with input data and configuration provided at runtime
   via a configuration file. */

/* Compared to the "normal" wave_reader, this code is specialized
   to two active tracers (heat and salt content) plus one optional
   passive tracer (dye) */

#include "../Science.hpp"
#include "../TArray.hpp"
#include "../Par_util.hpp"
#include "../NSIntegrator.hpp"
#include "../BaseCase.hpp"
#include "../Options.hpp"
#include <stdio.h>
#include <mpi.h>
#include <vector>
#include <random/uniform.h>
#include <random/normal.h>
#include <string>

using std::string;


int          Nx, Ny, Nz; // Number of points in x, y, z
double       Lx, Ly, Lz; // Grid lengths of x, y, z 
double   MinX, MinY, MinZ; // Minimum x/y/z points


// Input file names

string xgrid_filename,
       ygrid_filename,
       zgrid_filename,
       u_filename,
       v_filename,
       w_filename,
       T_filename,
       S_filename,
       dye_filename;

// Physical parameters
double g, rot_f, vel_mu, kappa_T, kappa_S,kappa_dye;

// Writeout parameters
double final_time, plot_interval;

double initial_time;

// Mapped grid?
bool mapped;

// Passive tracer?
bool tracer;

// Grid types
DIMTYPE intype_x, intype_y, intype_z;

static enum {
   MATLAB,
   CTYPE,
   FULL3D
} input_data_types;

using std::vector;

blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

using ranlib::Normal;

int myrank = -1;

bool restarting = false;
double restart_time = 0;
int restart_sequence = -1;

double perturb = 0;


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
         if (t == 0) return kappa_T; 
         if (t == 1) return kappa_S;
         if (t == 2) return kappa_dye;
         else assert(0 && "Invalid tracer number!");
      }

      double init_time() const { 
         return initial_time;
      }

      bool is_mapped() const {return mapped;}
      void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
         if (input_data_types == MATLAB) {
            if (master())
               fprintf(stderr,"Reading MATLAB-format xgrid (%d x %d) from %s\n",
                     Nx,Nz,xgrid_filename.c_str());
            read_2d_slice(xg,xgrid_filename.c_str(),Nx,Nz);
            if (master())
               fprintf(stderr,"Reading MATLAB-format zgrid (%d x %d) from %s\n",
                     Nx,Nz,zgrid_filename.c_str());
            read_2d_slice(zg,zgrid_filename.c_str(),Nx,Nz);
         } else if (input_data_types == CTYPE ||
                    input_data_types == FULL3D) {
            if (master())
               fprintf(stderr,"Reading CTYPE-format xgrid (%d x %d) from %s\n",
                     Nx,Nz,xgrid_filename.c_str());
            read_2d_restart(xg,xgrid_filename.c_str(),Nx,Nz);
            if (master())
               fprintf(stderr,"Reading CTYPE-format zgrid (%d x %d) from %s\n",
                     Nx,Nz,zgrid_filename.c_str());
            read_2d_restart(zg,zgrid_filename.c_str(),Nx,Nz);
         }
         // Automatically generate y-grid
         yg = 0*ii + MinY + Ly*(0.5+jj)/Ny + 0*kk;

         // Write out the grids and matlab readers
         write_array(xg,"xgrid");write_reader(xg,"xgrid",false);
         if (Ny > 1)
            write_array(yg,"ygrid"); write_reader(yg,"ygrid",false);
         write_array(zg,"zgrid"); write_reader(zg,"zgrid",false);
      }

      /* We have two active tracers, namely T and S */
      int numActive() const { return 2; }

      /* We're potentially given a passive tracer to advect */
      int numPassive() const {
         if (tracer) return 1;
         else return 0;
      }

      /* Timestep-check function.  This (long with "write everything" outputs) should
         really be bumped into the BaseCase */
      double check_timestep(double intime, double now) {
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
            write_array(*tracer[0],"T",plotnum);
            write_array(*tracer[1],"S",plotnum);
            // If we have a dye constituent, write it out
            if (tracer.size() > 2) {
               write_array(*tracer[2],"dye",plotnum);
            }
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
      }

      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         // Initialize the velocities from read-in data
         if (master()) fprintf(stderr,"Initializing velocities\n");
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* u */
            snprintf(filename,100,"u.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading u from %s\n",filename);
            read_array(u,filename,Nx,Ny,Nz);

            /* v, only necessary if this is an actual 3D run or if
               rotation is noNzero */
            if (Ny > 1 || rot_f != 0) {
               snprintf(filename,100,"v.%d",restart_sequence);
               if (master()) fprintf(stdout,"Reading v from %s\n",filename);
               read_array(v,filename,Nx,Ny,Nz);
            }

            /* w */
            snprintf(filename,100,"w.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading w from %s\n",filename);
            read_array(w,filename,Nx,Ny,Nz);
            return;
         }

         // Read in the appropriate data types
         switch(input_data_types) {
            case MATLAB: // MATLAB data
               if (master())
                  fprintf(stderr,"reading matlab-type u (%d x %d) from %s\n",
                        Nx,Nz,u_filename.c_str());
               read_2d_slice(u,u_filename.c_str(),Nx,Nz);
               if (v_filename != "" && (Ny >> 1 || rot_f != 0)) {
                  if (master())
                     fprintf(stderr,"reading matlab-type v (%d x %d) from %s\n",
                           Nx,Nz,v_filename.c_str());
                  read_2d_slice(v,v_filename.c_str(),Nx,Nz);
               } else {
                  v = 0;
               }
               if (master())
                  fprintf(stderr,"reading matlab-type w (%d x %d) from %s\n",
                        Nx,Nz,w_filename.c_str());
               read_2d_slice(w,w_filename.c_str(),Nx,Nz);
               break;
            case CTYPE: // Column-major 2D data
               if (master())
                  fprintf(stderr,"reading ctype u (%d x %d) from %s\n",
                        Nx,Nz,u_filename.c_str());
               read_2d_restart(u,u_filename.c_str(),Nx,Nz);
               if (v_filename != "" && (Ny >> 1 || rot_f != 0)) {
                  if (master())
                     fprintf(stderr,"reading ctype v (%d x %d) from %s\n",
                           Nx,Nz,v_filename.c_str());
                  read_2d_restart(v,v_filename.c_str(),Nx,Nz);
               } else {
                  v = 0;
               }
               if (master())
                  fprintf(stderr,"reading ctype w (%d x %d) from %s\n",
                        Nx,Nz,w_filename.c_str());
               read_2d_restart(w,w_filename.c_str(),Nx,Nz);
               break;
            case FULL3D:
               if (master()) 
                  fprintf(stdout,"Reading u from %s\n",
                     u_filename.c_str());
               read_array(u,u_filename.c_str(),Nx,Ny,Nz);
               if (master()) 
                  fprintf(stdout,"Reading u from %s\n",
                     v_filename.c_str());
               read_array(v,v_filename.c_str(),Nx,Ny,Nz);
               if (master()) 
                  fprintf(stdout,"Reading w from %s\n",
                     w_filename.c_str());
               read_array(w,w_filename.c_str(),Nx,Ny,Nz);
               break;
         }


         // Add a random perturbation to trigger any 3D instabilities
         if (perturb > 0) {
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
            Normal<double> rnd(0,1);
            for (int i = u.lbound(firstDim); i <= u.ubound(firstDim); i++) {
               rnd.seed(i);
               for (int j = u.lbound(secondDim); j <= u.ubound(secondDim); j++) {
                  for (int k = u.lbound(thirdDim); k <= u.ubound(thirdDim); k++) {
                     u(i,j,k) *= 1+perturb*rnd.random();
                     v(i,j,k) *= 1+perturb*rnd.random();
                     w(i,j,k) *= 1+perturb*rnd.random();
                  }
               }
            }
         }
         /* Write out initial values */
         write_array(u,"u",plotnum);
         write_reader(u,"u",true);
         if (Ny > 1 || rot_f != 0) {
            write_array(v,"v",plotnum);
            write_reader(v,"v",true);
         }
         write_array(w,"w",plotnum);
         write_reader(w,"w",true);
      }
      // Since we know we have at least two tracers (T and S), we can use
      // the full init_tracers(vector<DTArray *> & tracers) to initialize
      // everything at once.

      void init_tracers(vector<DTArray *> & in_tracers) {
         /* We'll define tracer number 0 to be T, tracer number 1 to be S,
            and tracer number 2 (if present) to be passive */

         if (master()) fprintf(stderr,"Initializing tracers\n");
         // First, make sure that the vector we're given has the expected size
         assert(numtracers() == int(in_tracers.size()));
         assert(numtracers() >= 2);

         // Define some convenient references
         DTArray & T = *in_tracers[0];
         DTArray & S = *in_tracers[1];
         // Since the "dye" may not always be present, we will -not- define
         // a convenient reference here.  That will act as a reminder to check
         // numtracers() at appropriate points.

         /* Initialize the tracers and take the opportunity to write out the grid */
         if (restarting) {
            /* Restarting, so build the proper filenames and load
               the data into u, v, w */
            char filename[100];
            /* T */
            snprintf(filename,100,"T.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading T from %s\n",filename);
            read_array(T,filename,Nx,Ny,Nz);
            snprintf(filename,100,"S.%d",restart_sequence);
            if (master()) fprintf(stdout,"Reading S from %s\n",filename);
            read_array(S,filename,Nx,Ny,Nz);
            // More than two tracers means we have dye
            if (numtracers() > 2) { 
               snprintf(filename,100,"dye.%d",restart_sequence);
               if (master()) fprintf(stdout,"Reading dye from %s\n",filename);
               read_array(*in_tracers[2],filename,Nx,Ny,Nz);
            }
            // If we're restarting, then the job's done -- the grid files and
            // readers are presumed to already exist.
            return;
         }
         // If we're not restarting, we have to call different reader functions
         // depending on the format of input data.
         switch (input_data_types) {
            case MATLAB:
               if (master())
                  fprintf(stderr,"reading matlab-type T (%d x %d) from %s\n",
                        Nx,Nz,T_filename.c_str());
               read_2d_slice(T,T_filename.c_str(),Nx,Nz);
               if (master())
                  fprintf(stderr,"reading matlab-type S (%d x %d) from %s\n",
                        Nx,Nz,S_filename.c_str());
               read_2d_slice(S,S_filename.c_str(),Nx,Nz);
               if (numtracers() > 2) {
                  if (master())
                     fprintf(stderr,"reading matlab-type dye (%d x %d) from %s\n",
                           Nx,Nz,dye_filename.c_str());
                  read_2d_slice(*in_tracers[2],dye_filename.c_str(),Nx,Nz);
               }
               break;
            case CTYPE:
               if (master())
                  fprintf(stderr,"reading ctype-type T (%d x %d) from %s\n",
                        Nx,Nz,T_filename.c_str());
               read_2d_restart(T,T_filename.c_str(),Nx,Nz);
               if (master())
                  fprintf(stderr,"reading ctype-type S (%d x %d) from %s\n",
                        Nx,Nz,S_filename.c_str());
               read_2d_restart(S,S_filename.c_str(),Nx,Nz);
               if (numtracers() > 2) {
                  if (master())
                     fprintf(stderr,"reading ctype-type dye (%d x %d) from %s\n",
                           Nx,Nz,dye_filename.c_str());
                  read_2d_restart(*in_tracers[2],dye_filename.c_str(),Nx,Nz);
               }
               break;
            case FULL3D:
               if (master())
                  fprintf(stderr,"reading 3D T (%d x %d x %d) from %s\n",
                        Nx,Ny,Nz,T_filename.c_str());
               read_array(T,T_filename.c_str(),Nx,Ny,Nz);
                  fprintf(stderr,"reading 3D S (%d x %d x %d) from %s\n",
                        Nx,Ny,Nz,S_filename.c_str());
               read_array(S,S_filename.c_str(),Nx,Ny,Nz);
               if (numtracers() > 2) {
                  if (master())
                     fprintf(stderr,"reading 3D dye (%d x %d x %d) from %s\n",
                           Nx,Ny,Nz,dye_filename.c_str());
                  read_array(*in_tracers[3],dye_filename.c_str(),Nx,Ny,Nz);
               }
               break;
         }
         write_array(T,"T",plotnum);
         write_array(S,"S",plotnum);
         write_reader(T,"T",true);
         write_reader(S,"S",true);
         if (numtracers() > 2) {
            write_array(*in_tracers[2],"dye",plotnum);
            write_reader(*in_tracers[2],"dye",true);
         }
         /* Write out the pressure reader, but we don't have a pressure field
            yet; use T as a proxy because it will have the same size and memory
            shape as pressure */
         write_reader(T,"p",true);
      }

      void forcing(double t, DTArray & u, DTArray & u_f,
          DTArray & v, DTArray & v_f, DTArray & w, DTArray & w_f,
          vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
         DTArray & heat = *tracers[0];
         DTArray & salt = *tracers[1];
         /* Velocity forcing */
         u_f = -rot_f * v; v_f = +rot_f * u;
         w_f = -g*eqn_of_state(heat,salt)/1000.0;// rho_0 is 1000.0
         /* Heat and Salt forcing */
         *(tracers_f[0]) = 0;
         *(tracers_f[1]) = 0;
         // If it exists, there is zero forcing for dye
         if (numtracers() > 2)
            *(tracers_f[2]) = 0;
      }

      userControl() :
         plotnum(restart_sequence), 
         nextplot(initial_time + plot_interval), itercount(0), lastplot(0),
         last_writeout(0) 
         {
            compute_quadweights(size_x(),size_y(),size_z(),
                  length_x(),length_y(),length_z(),
                  type_x(),type_y(),type_z());
            // If this is an unmapped grid, generate/write the
            // 3D grid files
            if (!is_mapped()) {
               automatic_grid(MinX,MinY,MinZ);
            }
         }
};

int main(int argc, char ** argv) {
   /* Set implicit filtering */
   //f_strength = -.25;
   //f_order = 4;
   //f_cutoff = 0.6;
   MPI_Init(&argc,&argv);

   // To properly handle the variety of options, set up the boost
   // program_options library using the abbreviated interface in
   // ../Options.hpp

   options_init(); // Initialize options

   option_category("Grid Options");

   add_option("Nx",&Nx,"Number of points in X");
   add_option("Ny",&Ny,1,"Number of points in Y");
   add_option("Nz",&Nz,"Number of points in Z");

   string xgrid_type, ygrid_type, zgrid_type;
   add_option("type_x",&xgrid_type,
         "Grid type in X.  Valid values are:\n"
         "   FOURIER: Periodic\n"
         "   FREE_SLIP: Cosine expansion\n"
         "   NO_SLIP: Chebyhsev expansion");
   add_option("type_y",&ygrid_type,"FOURIER","Grid type in Y");
   add_option("type_z",&zgrid_type,"Grid type in Z");

   add_option("Lx",&Lx,"X-length");
   add_option("Ly",&Ly,1.0,"Y-length");
   add_option("Lz",&Lz,"Z-length");

   add_option("min_x",&MinX,0.0,"Unmapped grids: Minimum X-value");
   add_option("min_y",&MinY,0.0,"Minimum Y-value");
   add_option("min_z",&MinZ,0.0,"Minimum Z-value");

   option_category("Grid mapping options");
   add_option("mapped_grid",&mapped,false,"Use a mapped (2D) grid");
   add_option("xgrid",&xgrid_filename,"x-grid filename");
//   add_option("ygrid",&ygrid_filename,"","y-grid filename");
   add_option("zgrid",&zgrid_filename,"z-grid filename");

   option_category("Input data");
   string datatype;
   add_option("file_type",&datatype,
         "Format of input data files, including that for the mapped grid."
         "Valid options are:\n"
         "   MATLAB: \tRow-major 2D arrays of size Nx x Nz\n"
         "   CTYPE:  \tColumn-major 2D arrays (including that output by 2D SPINS runs)\n"
         "   FULL:   \tColumn-major 3D arrays; implies CTYPE for grid mapping if enabled");

   add_option("u_fila",&u_filename,"U-velocity filename");
   add_option("v_file",&v_filename,"","V-velocity filename");
   add_option("w_file",&w_filename,"W-velocity filename");
   add_option("T_file",&T_filename,"T (temperature) filename");
   add_option("S_file",&T_filename,"S (salinity) filename");

   option_category("Passive tracer");
   add_switch("enable_dye",&tracer,"Enable evolution of a passive tracer (dye)");
   add_option("dye_file",&dye_filename,"Dye filename");
   add_option("kappa_dye",&kappa_dye,"Diffusivity of dye");

   option_category("Physical parameters");
   add_option("g",&g,9.81,"Gravitational acceleration");
   add_option("rot_f",&rot_f,0.0,"Coriolis force term");
   add_option("visc",&vel_mu,0.0,"Kinematic viscosity");
   add_option("kappa_T",&kappa_T,0.0,"Thermal diffusivity");
   add_option("kappa_S",&kappa_S,0.0,"Salt diffusivity");

   add_option("perturbation",&perturb,0.0,"Veloc\tity perturbation (multiplicative white noise) applied to read-in data.");

   option_category("Running options");
   add_option("init_time",&initial_time,0.0,"Initial time");
   add_option("final_time",&final_time,"Final time");
   add_option("plot_interval",&plot_interval,"Interval between output times");

   option_category("Restart options");
   add_switch("restart",&restarting,"Restart from prior output time.  OVERRIDES many other values.");
   add_option("restart_time",&restart_time,0.0,"Time to restart from");
   add_option("restart_sequence",&restart_sequence,
         "Sequence number to restart from (if plot_interval has changed)");


   // Parse the options from the command line and config file
   options_parse(argc,argv);

   // Now, make sense of the options received.  Many of these values
   // can be directly used, but the ones of string-type need further
   // procesing.

   // Grid types:

   if (xgrid_type == "FOURIER") {
      intype_x = PERIODIC;
   } else if (xgrid_type == "FREE_SLIP") {
      intype_x = FREE_SLIP;
   } else if (xgrid_type == "NO_SLIP") {
      intype_x = NO_SLIP;
   } else {
      if (master())
         fprintf(stderr,"Invalid option %s received for type_x\n",xgrid_type.c_str());
      MPI_Finalize(); exit(1);
   }
   if (ygrid_type == "FOURIER") {
      intype_y = PERIODIC;
   } else if (ygrid_type == "FREE_SLIP") {
      intype_y = FREE_SLIP;
   } else {
      if (master())
         fprintf(stderr,"Invalid option %s received for type_y\n",ygrid_type.c_str());
      MPI_Finalize(); exit(1);
   }
   if (zgrid_type == "FOURIER") {
      intype_z = PERIODIC;
   } else if (zgrid_type == "FREE_SLIP") {
      intype_z = FREE_SLIP;
   } else if (zgrid_type == "NO_SLIP") {
      intype_z = NO_SLIP;
   } else {
      if (master())
         fprintf(stderr,"Invalid option %s received for type_z\n",zgrid_type.c_str());
      MPI_Finalize(); exit(1);
   }

   // Input filetypes

   if (datatype=="MATLAB") {
      input_data_types = MATLAB;
   } else if (datatype == "CTYPE") {
      input_data_types = CTYPE;
   } else if (datatype == "FULL") {
      input_data_types = FULL3D;
   } else {
      if (master())
         fprintf(stderr,"Invalid option %s received for file_type\n",datatype.c_str());
      MPI_Finalize(); exit(1);
   }

   if (restarting) {
      if (restart_sequence <= 0) {
         restart_sequence = int(restart_time/plot_interval);
      }
      if (master()) {
         fprintf(stderr,"Restart flags detected\n");
         fprintf(stderr,"Restarting from time %g, at sequence number %d\n",
               restart_time,restart_sequence);
      }
      initial_time = restart_time;
   } else {
      // Not restarting, so set the initial sequence number
      // to the initial time / plot_interval
      restart_sequence = int(initial_time/plot_interval);
      if (fmod(initial_time,plot_interval) != 0.0) {
         if (master()) {
            fprintf(stderr,"Warning: the initial time (%g) does not appear to be an even multiple of the plot interval (%g)\n",
                  initial_time,plot_interval);
         }
      }
   }
   userControl mycode;
   FluidEvolve<userControl> kevin_kh(&mycode);
   kevin_kh.initialize();
   kevin_kh.do_run(final_time);
   MPI_Finalize();
}
