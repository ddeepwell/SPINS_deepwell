/* Generic script for two layer fluid (tanh profile)
   with zero initial velocity
   and no topography */

// Required headers
#include "../TArray.hpp"        // Custom extensions to the library to support FFTs
#include "../NSIntegrator.hpp"  // Time-integrator for the Navier-Stokes equations
#include "../BaseCase.hpp"      // Support file that contains default implementations of several functions
#include "../Options.hpp"       // config-file parser
#include <random/normal.h>      // Blitz random number generator
#include <blitz/array.h>        // Blitz++ array library
#include <mpi.h>                // MPI parallel library
#include <iostream>
#include <fstream>
#include <stdlib.h>
//#include "../Science.hpp"       // Additional analysis routines

using namespace std;
using namespace NSIntegrator;
using namespace ranlib;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;


/* ------------------ Parameters --------------------- */
// Grid scales
double Lx, Ly, Lz;              // (m)
int Nx, Ny, Nz;                 // Points in x, y (span), z directions
double MinX, MinY, MinZ;        // Minimum x/y/z points
// Grid types
DIMTYPE intype_x, intype_y, intype_z;

// Physical constants
double g, ROT_F;                // gravity accel (m/s^2), Coriolis frequency (s^-1)

// Stratification parameters
double rho_0;                   // reference density (kg/L)
double delta_rho;               // density difference between top and bottom layers (kg/L)
double pyc_asym;                // % of depth to shift pycnocline above the mid-depth
double h_perc;                  // pycnocline half-width as % of depth
double h_mix_perc;              // vertical half-width transition of mixed region
// Horizontal stratification parameters
double delta_x;                 // horizontal transition length (m)
double Lmix;                    // Width of mixed region (m)
double Hmix;                    // Total height of mixed region (m)

// Viscosity and diffusivity of density and tracers
double VISCO;
double DIFFU_rho;
double DIFFU_dye_1;

// Temporal parameters
double plot_interval;           // Time between field writes (s)
double final_time;              // Final time (s)

// Vertical chain parameters
bool savechain;                 // (boolean) Flag to use chain or not
double chain1_start_time;       // time to start saving chain (s)
double chain1_end_time;         // time to stop saving chain (s)
double chain_plot_interval;                     // time between chain writes (s)
// Chain locations (defined in main program)
int chain1_xindex, chain1_yindex;

// Initial velocity perturbation
double u0_pert;

// iteration coutner
int itercount = 0;

// Dump parameters
double real_start_time;
double compute_time;
bool restart_from_dump = false;
double total_run_time;
double avg_write_time;

// Restarting options (these are defaults)
bool restarting = false;
double restart_time = 0;
double initial_time = 0;
int restart_sequence = -1;

/* ------------------ Derived parameters --------------------- */

// Pycnocline half-width
double h_halfwidth;
double h_mix_half;

// Diffusivity information
const int N_tracers = 2;        // must be self defined, and add one for density
double DIFFU[ N_tracers ];
double *DIFFU_pointer;

// Flow speed
double c0;

// Squared maximum buoyancy frequency if the initial stratification was stable
double N2_max;

// Reynolds number
double Re;


/* ------------------ Initialize dambreak class --------------------- */

class dambreak : public BaseCase {
    public:
        // Arrays for 1D grids defined here
        Array<double,1> xx, yy,zz;

        // Helper variables for the plot number and time of last plotting
        int plot_number;
        double last_plot, chain_last_plot;
        // variables for timing steps
        double t_step;
        double clock_time, step_start_time;

        /* The grid size is governed through the definitions above */
        double length_x() const { return Lx;}
        double length_y() const { return Ly;}
        double length_z() const { return Lz;}

        // Resolution in X, Y, and Z
        int size_x() const { return Nx; }
        int size_y() const { return Ny; }
        int size_z() const { return Nz; }

        /* Set expansions (FREE_SLIP, NO_SLIP (in vertical) or PERIODIC) */
        DIMTYPE type_x() const { return intype_x; }
        DIMTYPE type_y() const { return intype_y; }
        DIMTYPE type_z() const { return intype_z; }    // Check to confirm with the definition of zz below!
        DIMTYPE type_default() const { return PERIODIC; }

        /* Number of tracers */
        // the first is density
        int numtracers() const { return N_tracers; }

        /* Viscosity and diffusivity */
        double get_visco() const { return VISCO; }
        double get_diffusivity(int t_num) const {
            return DIFFU[ t_num ];
        }

        /* Initial time */
        double init_time() const { return initial_time; }

        /* Modify the timestep if necessary in order to land evenly on a plot time */
        double check_timestep (double intime, double now) {
            // Firstly, the buoyancy frequency provides a timescale that is not
            // accounted for with the velocity-based CFL condition.
            if (intime > 0.5/sqrt(N2_max)) {
                intime = 0.5/sqrt(N2_max);
            }
            // Now, calculate how many timesteps remain until the next writeout
            double until_plot = last_plot + plot_interval - now;
            int steps = ceil(until_plot / intime);
            // And calculate where we will actually be after (steps) timesteps
            // of the current size
            double true_fintime = steps*intime;

            // If that's close enough to the real writeout time, that's fine.
            if (fabs(until_plot - true_fintime) < 1e-6) {
                return intime;
            } else {
                // Otherwise, square up the timeteps.  This will always shrink the timestep.
                return (until_plot / steps);
            }
        }


        /* Initialize velocities at the start of the run.  For this simple
           case, initialize all velocities to 0 */
        void init_vels(DTArray & u, DTArray & v, DTArray & w) {
            if (restarting and (!restart_from_dump)) {
                init_vels_restart(u, v, w);
            }
            else if (restarting and restart_from_dump) {
                init_vels_dump(u, v, w);
            }
            else{
                u = 0; // Use the Blitz++ syntax for simple initialization
                v = 0; // of an entire (2D or 3D) array with a single line
                w = 0; // of code.
                /* Add random initial perturbation */
                int myrank;
                MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
                /* Add random noise about 3 orders of magnitude below dipole */
                Normal<double> rnd(0,1);
                rnd.seed(myrank);
                for (int i = u.lbound(firstDim); i<= u.ubound(firstDim); i++) {
                    for (int j = u.lbound(secondDim); j<= u.ubound(secondDim); j++) {
                        for (int k = u.lbound(thirdDim); k<= u.ubound(thirdDim); k++) {
                            u(i,j,k) += u0_pert*rnd.random();
                            v(i,j,k) += u0_pert*rnd.random();
                            w(i,j,k) += u0_pert*rnd.random();
                        }
                    }
                }

                // Also, write out the initial velocities and proper M-file readers
                write_reader(u,"u",true);
                write_reader(w,"w",true);
                write_array(u,"u",plot_number);
                write_array(w,"w",plot_number);
                if (Ny > 1 || ROT_F != 0) {
                    write_reader(v,"v",true);
                    write_array(v,"v",plot_number);
                }
                return;
            }
        }

        /* Initialze the tracers (density and dye) */
        void init_tracer(int t_num, DTArray & the_tracer) {
            /* Initialize the density */
            if (t_num == 0) {
                if (restarting and (!restart_from_dump)) {
                    init_tracer_restart("rho",the_tracer);
                }
                else if (restarting and restart_from_dump) {
                    init_tracer_dump("rho",the_tracer);
                }
                else {
                    if (master()) fprintf(stdout,"Initializing tracer %d\n",t_num);

                    // background stratification
                    the_tracer = -0.5*delta_rho*tanh((zz(kk)-(0.5+pyc_asym)*Lz)/h_halfwidth);
                    the_tracer = the_tracer*0.5*(1.0+tanh((xx(ii)-Lmix)/delta_x));
                    // mixed region
                    the_tracer = the_tracer + 0.5*(1.0-tanh((xx(ii)-Lmix)/delta_x))
                        *(-0.25*delta_rho)*(
                                1.0+tanh((zz(kk)-(0.5+pyc_asym+0.5*Hmix)*Lz)/h_mix_half)
                                -1.0+tanh((zz(kk)-(0.5+pyc_asym-0.5*Hmix)*Lz)/h_mix_half));
                    write_array(the_tracer,"rho",plot_number);
                    write_reader(the_tracer,"rho",true);
                }
            }
            /* Tracer 1 */
            else if (t_num == 1) {
                if (restarting and (!restart_from_dump)) {
                    init_tracer_restart("dye1",the_tracer);
                }
                else if (restarting and restart_from_dump) {
                    init_tracer_dump("dye1",the_tracer);
                }
                else {
                    if (master()) fprintf(stdout,"Initializing tracer %d\n",t_num);

                    the_tracer = 1.0-0.5*(1.0+tanh((xx(ii)-Lmix)/delta_x));
                    write_array(the_tracer,"dye1",plot_number);
                    write_reader(the_tracer,"dye1",true);
                }
            }
        }

        /* Forcing in the momentum equations */
        void forcing(double t, const DTArray & u, DTArray & u_f,
                const DTArray & v, DTArray & v_f, const DTArray & w,
                DTArray & w_f, vector<DTArray *> & tracers,
                vector<DTArray *> & tracers_f) {
            u_f = -ROT_F*v;
            v_f = +ROT_F*u;
            w_f = -g*((*tracers[0]))/rho_0;
            *tracers_f[0] = 0;
            *tracers_f[1] = 0; // The passive tracer also has no forcing
        }

        /* Basic analysis, to write out the field periodically */
        void analysis(double sim_time, DTArray & u, DTArray & v, DTArray & w,
                vector<DTArray *> & tracer, DTArray & pressure) {
            /* If it is very close to the plot time, write data fields to disk */
            if ((sim_time - last_plot - plot_interval) > -1e-6) {
                plot_number++;
                t_step = MPI_Wtime(); // time just before write (for dump)
                write_array(u,"u",plot_number);
                write_array(w,"w",plot_number);
                if (Ny > 1 || ROT_F != 0) {
                    write_array(v,"v",plot_number);
                }
                write_array(*tracer[0],"rho",plot_number);
                write_array(*tracer[1],"dye1",plot_number);
                last_plot = last_plot + plot_interval;

                // Find average time to write (for dump)
                clock_time = MPI_Wtime(); // time just afer write
                avg_write_time = (avg_write_time*(plot_number-restart_sequence-1) + (clock_time - t_step))/
                    (plot_number-restart_sequence);
                if (master()){
                    fprintf(stdout,"Last write time: %.6g. Average write time: %.6g.\n", clock_time - t_step, avg_write_time);
                }
                if (master()) fprintf(stdout,"*");
            }
            // increase counter and update clocks
            itercount++;
            if (itercount == 1){
                step_start_time = MPI_Wtime();
            }
            if (master()) {
                clock_time = MPI_Wtime();
                t_step = clock_time - step_start_time;
            }

            // Also, calculate and write out useful information: maximum u, v, w...
            double max_u = psmax(max(abs(u)));
            double max_v = psmax(max(abs(v)));
            double max_w = psmax(max(abs(w)));
            double max_ke = psmax(max(0.5*rho_0*(u*u + v*v + w*w)*Lx/Nx*Ly/Ny*Lz/Nz)); //only true for uniform grid
            double ke_tot = pssum(sum(0.5*rho_0*(u*u + v*v + w*w)*Lx/Nx*Ly/Ny*Lz/Nz));  //only true for uniform grid
            double max_rho = psmax(max(abs(*tracer[0])));
            double max_dye1 = psmax(max(abs(*tracer[1])));
            if (master() && itercount == 1){
                double t_startup;
                t_startup = clock_time - real_start_time;
                fprintf(stdout,"Start-up time: %g s.\n",t_startup);
            }
            if (master() && itercount == 1) fprintf(stdout,"[Iter], (Clock time), Sim time:, Max U, Max V, Max W, Max KE, Total KE, Max Density, Max Dye 1\n");
            if (master()) fprintf(stdout,"[%d] (%.6g) %.6f: %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                    itercount,t_step,sim_time,max_u,max_v,max_w,max_ke,ke_tot,max_rho,max_dye1);

            if (savechain){
                // Also, write out vertical chains of data
                int Iout,Jout,myrank,pointflag;
                MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                // All other fields
                if ((sim_time >= chain1_start_time)
                        && ((sim_time - chain_last_plot - chain_plot_interval) > -1e-6)
                        && (sim_time < chain1_end_time)) {
                    Iout = chain1_xindex; Jout = chain1_yindex; pointflag = 0;
                    if ( Iout >= u.lbound(firstDim) && Iout <= u.ubound(firstDim) ) pointflag=1;
                    if (pointflag==1) {
                        writechain("u_chain.txt",u, Iout, Jout, sim_time);
                        writechain("v_chain.txt",v, Iout, Jout, sim_time);
                        writechain("w_chain.txt",w, Iout, Jout, sim_time);
                        writechain("rho_chain.txt",*tracer[0], Iout, Jout, sim_time);
                        writechain("dye1_chain.txt",*tracer[1], Iout, Jout, sim_time);
                    }
                    chain_last_plot = chain_last_plot + chain_plot_interval;
                }
            }

            // Determine last plot if restarting from the dump case
            if (restart_from_dump and (itercount == 1)){
                last_plot = restart_sequence*plot_interval;    
            }
            // see if close to end of compute time and dump
            check_and_dump(clock_time, real_start_time, compute_time, sim_time, avg_write_time,
                    plot_number, u, v, w, tracer);
            // Change dump log file if successfully reached final time
            // the dump time will be twice final time so that a restart won't actually start
            successful_dump(plot_number, final_time, plot_interval);
        }

        void writechain(const char *filename, DTArray & val, int Iout, int Jout, double sim_time) {
            FILE *fid=fopen(filename,"a");
            if (fid==NULL) {
                fprintf(stderr,"Unable to open %s for writing\n",filename);
                exit(1);
            }
            fprintf(fid,"%g",sim_time);
            for (int ki=0; ki<Nz; ki++) fprintf(fid," %g",val(Iout,Jout,ki));
            fprintf(fid,"\n");
            fclose(fid);
        }

        // User specified variables to dump
        void write_variables(DTArray & u,DTArray & v, DTArray & w,
                vector<DTArray *> & tracer) {
            write_array(u,"u.dump",-1);
            write_array(v,"v.dump",-1);
            write_array(w,"w.dump",-1);
            write_array(*tracer[0],"rho.dump",-1);
            write_array(*tracer[1],"dye1.dump",-1);
        }

        // Constructor
        dambreak(): // Initialization list for xx, yy and zz 1d grids
            xx(split_range(Nx)), yy(Ny), zz(Nz)
    {   // Initialize the local variables
        plot_number = restart_sequence;
        last_plot = restart_time;
        chain_last_plot = chain1_start_time - chain_plot_interval;
        // Create one-dimensional arrays for the coordinates
        if (type_x() == NO_SLIP) {
            xx = MinX + Lx*(0.5+0.5*cos(M_PI*ii/(Nx-1)));
        } else {
            xx = MinX + Lx*(ii+0.5)/Nx;
        }
        yy = MinY + Ly*(ii+0.5)/Ny;
        if (type_z() == NO_SLIP) {
            zz = MinZ + Lz*(0.5+0.5*cos(M_PI*ii/(Nz-1)));
        } else {
            zz = MinZ + Lz*(0.5+ii)/Nz;
        }
        automatic_grid(MinX,MinY,MinZ);
    }
};


/* The ``main'' routine */
int main(int argc, char ** argv) {
    /* Initialize MPI.  This is required even for single-processor runs,
       since the inner routines assume some degree of parallelization,
       even if it is trivial. */
    MPI_Init(&argc, &argv);

    real_start_time = MPI_Wtime();     // for dump
    /* ------------------ Define parameters from spins.conf --------------------- */
    options_init(); // Initialize options

    option_category("Restart options");
    add_switch("restart",&restarting,"Restart from prior output time.  OVERRIDES many other values.");
    add_option("restart_time",&restart_time,0.0,"Time to restart from");
    add_option("restart_sequence",&restart_sequence,"Sequence number to restart from");

    option_category("Grid Options");
    add_option("Lx",&Lx,"Length of tank");
    add_option("Ly",&Ly,"Width of tank");
    add_option("Lz",&Lz,"Height of tank");
    add_option("Nx",&Nx,"Number of points in X");
    add_option("Ny",&Ny,1,"Number of points in Y");
    add_option("Nz",&Nz,"Number of points in Z");
    add_option("min_x",&MinX,"Minimum X-value");
    add_option("min_y",&MinY,"Minimum Y-value");
    add_option("min_z",&MinZ,"Minimum Z-value");

    string xgrid_type, ygrid_type, zgrid_type;
    add_option("type_x",&xgrid_type,
            "Grid type in X.  Valid values are:\n"
            "   FOURIER: Periodic\n"
            "   FREE_SLIP: Cosine expansion\n"
            "   NO_SLIP: Chebyhsev expansion");
    add_option("type_y",&ygrid_type,"FOURIER","Grid type in Y");
    add_option("type_z",&zgrid_type,"Grid type in Z");

    add_option("g",&g,9.81,"Gravitational acceleration");
    add_option("ROT_F",&ROT_F,0.0,"Coriolis frequency");
    add_option("rho_0",&rho_0,"Reference density");
    add_option("delta_rho",&delta_rho,"Density difference b/w top and bottom layers");
    add_option("h_perc",&h_perc,"Pycnocline half-width as perc. of depth");
    add_option("h_mix_perc",&h_mix_perc,"Pycnocline half-width as perc. of depth");
    add_option("pyc_asym",&pyc_asym,"percentage of depth to shift pycnocline");
    add_option("delta_x",&delta_x,"Horizontal transition half-width");
    add_option("Lmix",&Lmix,"Width of mixed region");
    add_option("Hmix",&Hmix,"Width of mixed region");

    add_option("visco",&VISCO,"Viscosity");
    add_option("diffu_rho",&DIFFU_rho,"Diffusivity of density");
    add_option("diffu_dye_1",&DIFFU_dye_1,"Diffusivity of dye 1");	

    add_option("plot_interval",&plot_interval,"Time between writes");
    add_option("final_time",&final_time,"Final time");
    add_option("savechain",&savechain,false,"Flag to have save vertical chains or not");
    add_option("chain1_start_time",&chain1_start_time,"Time to start writing chain");
    add_option("chain1_end_time",&chain1_end_time,"Time to stop writing chain");
    add_option("chain_plot_interval",&chain_plot_interval,"Time between writes in chain");

    add_option("u0_pert",&u0_pert,"Initial perturbation in velocity");

    option_category("Dumping options");
    add_option("compute_time",&compute_time,-1.0,"Time permitted for computation");
    add_option("restart_from_dump",&restart_from_dump,false,"If restart from dump");

    options_parse(argc,argv);

    // Now, make sense of the options received.  Many of these values
    // can be directly used, but the ones of string-type need further
    // procesing.

    // Read dump_time.txt and check if past final time
    if (restart_from_dump){
        restarting = true;
        string dump_str;
        ifstream dump_file;
        dump_file.open ("dump_time.txt");

        getline (dump_file,dump_str); // ingnore 1st line

        getline (dump_file,dump_str);
        restart_time = atof(dump_str.c_str());

        getline (dump_file,dump_str); // ingore 3rd line

        getline (dump_file,dump_str);
        restart_sequence = atoi(dump_str.c_str());

        if (restart_time > final_time){
            // Die, ungracefully
            if (master()){
                fprintf(stderr,"Restart dump time (%.4g) is past final time (%.4g). Quitting now.\n",restart_time,final_time);
            }
            MPI_Finalize(); exit(1);
        }
    }
    if (compute_time > 0){
        avg_write_time = max(100.0*Nx*Ny*Nz/pow(512.0,3), 20.0);
    }

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

    /* ------------------ Derived parameters --------------------- */

    // Vertical chain locations
    chain1_xindex = Nx/2;
    chain1_yindex = Ny/2;

    // Pycnocline half-width
    h_halfwidth = h_perc*Lz;
    h_mix_half  = h_mix_perc*Lz;

    // Diffusivity information
    DIFFU_pointer = &DIFFU[0];  *DIFFU_pointer = DIFFU_rho;
    DIFFU_pointer = &DIFFU[1];  *DIFFU_pointer = DIFFU_dye_1;

    // Mode-2 wave speed
    c0 = sqrt(g*h_halfwidth*(0.5*delta_rho)/rho_0); // I'm not sure on this

    // Maximum buoyancy frequency (squared) if the initial stratification was stable
    N2_max = g/rho_0*delta_rho/(2*h_halfwidth);

    // Reynolds number
    Re = c0*h_halfwidth/VISCO;


    /* ------------------ Set correct initial time, and sequence --------------------- */
    if (restarting) {
        if (restart_sequence <= 0) {
            restart_sequence = int(restart_time/plot_interval);
        }
        if (master()) {
            fprintf(stdout,"Restart flags detected\n");
            fprintf(stdout,"Restarting from time %g, at sequence number %d\n",
                    restart_time,restart_sequence);
        }
        initial_time = restart_time;
    } else {
        // Not restarting, so set the initial sequence number
        // to the initial time / plot_interval
        restart_sequence = int(initial_time/plot_interval);
        if (fmod(initial_time,plot_interval) != 0.0) {
            if (master()) {
                fprintf(stdout,"Warning: the initial time (%g) does not appear to be an even multiple of the plot interval (%g)\n",
                        initial_time,plot_interval);
            }
        }
    }

    /* ------------------ Print some parameters --------------------- */
    if (master()) {
        fprintf(stdout,"Dam break problem\n");
        fprintf(stdout,"Using a %f x %f x %f grid of %d x %d x %d points\n",Lx,Ly,Lz,Nx,Ny,Nz);
        fprintf(stdout,"g = %f, ROT_F = %f, rho_0 = %f, delta_rho %f\n",g,ROT_F,rho_0,delta_rho);
        fprintf(stdout,"Pycnocline half-width as %% of depth: h_perc = %g\n",h_perc);
        fprintf(stdout,"Pycnocline half-width: h = %g\n",h_halfwidth);
        fprintf(stdout,"Pycnocline vertical shift %%: zeta = %g\n",pyc_asym);
        fprintf(stdout,"Width of mixed region: L_mix = %g\n",Lmix);
        fprintf(stdout,"Height of mixed region as %% of depth: H_mix = %g\n",Hmix);
        fprintf(stdout,"Time between plots: %g s\n",plot_interval);
        fprintf(stdout,"Chain 1 indices: x_i = %d, y_i = %d\n",chain1_xindex,chain1_yindex);
        fprintf(stdout,"Initial velocity perturbation: %g\n",u0_pert);

        fprintf(stdout,"Stably-stratified phase speed %g\n",c0);
        fprintf(stdout,"Buoyancy frequency squared %g\n",N2_max);
        fprintf(stdout,"Reynolds number %g\n",Re);
    }

    /* ------------------ Do stuff --------------------- */
    dambreak mycode; // Create an instantiated object of the above class
    /// Create a flow-evolver that takes its settings from the above class
    FluidEvolve<dambreak> do_stuff(&mycode);

    // Initialize
    do_stuff.initialize();

    // Run until the end of time
    do_stuff.do_run(final_time);
    MPI_Finalize(); // Cleanly exit MPI
    return 0; // End the program
}





