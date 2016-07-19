/* wave_reader.cpp -- general case for looking at the evolution of
   waves, with input data and configuration provided at runtime
   via a configuration file. */

/* ------------------ Top matter --------------------- */

// Required headers
#include "../Science.hpp"       // Additional analysis routines
#include "../TArray.hpp"        // Custom extensions to the library to support FFTs
#include "../Par_util.hpp"
#include "../NSIntegrator.hpp"  // Time-integrator for the Navier-Stokes equations
#include "../BaseCase.hpp"      // Support file that contains default implementations of many functions
#include "../Options.hpp"       // config-file parser
#include <random/normal.h>      // Blitz random number generator
#include <random/uniform.h>
#include <mpi.h>                // MPI parallel library
#include <vector>
#include <stdio.h>
#include <string>
#include <fstream>

using std::string;
using std::vector;
using ranlib::Normal;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

/* ------------------ Parameters --------------------- */

// Grid scales
double Lx, Ly, Lz;          // Grid lengths (m)
int    Nx, Ny, Nz;          // Number of points in x, y, z
double MinX, MinY, MinZ;    // Minimum x/y/z points
// Mapped grid?
bool mapped;
// Grid types
DIMTYPE intype_x, intype_y, intype_z;

// Physical parameters
double g, rot_f, rho_0;     // gravity accel (m/s^2), Coriolis frequency (s^-1), reference density (kg/L)
double VISCO;               // viscosity (m^2/s)
double kappa_rho;           // diffusivity of density (m^2/s)

// tracer options
static const int RHO = 0;   // index for rho
bool tracer;
double kappa_trc, tracer_g;

// Temporal parameters
double final_time;          // Final time (s)
double plot_interval;       // Time between field writes (s)
double dt_max;              // maximum time during a single time step

// Restarting options
bool restarting;            // are you restarting?
double restart_time;        // start time when restarting
double initial_time;        // start time when not restarting
int restart_sequence;

// Dump parameters
bool restart_from_dump;
double compute_time;
double real_start_time;
double total_run_time;
double avg_write_time;

// Write out pressure?
bool pressure_write;
// Initial velocity perturbation
double perturb;
// Iteration counter
int itercount = 0;

// Possible input data types
static enum {
    MATLAB,
    CTYPE,
    FULL3D
} input_data_types;

// Input file names
string xgrid_filename,
       ygrid_filename,
       zgrid_filename,
       u_filename,
       v_filename,
       w_filename,
       rho_filename,
       tracer_filename;


/* ------------------ Initialize the class --------------------- */

class userControl : public BaseCase {
    public:
        /* Grid arrays */
        DTArray *zgrid;

        /* Timing variables (for outputs and measuring time steps) */
        int plotnum;        // most recent output number (for plotting)
        double last_plot;   // most recent output time 
        double next_plot;   // time of next output write
        // variables for timing steps
        double t_step;
        double clock_time, step_start_time;

        /* Size of domain */
        double length_x() const { return Lx; }
        double length_y() const { return Ly; }
        double length_z() const { return Lz; }

        /* Resolution in X, Y, and Z */
        int size_x() const { return Nx; }
        int size_y() const { return Ny; }
        int size_z() const { return Nz; }

        /* Set expansion (FREE_SLIP, NO_SLIP (in vertical) or PERIODIC) */
        DIMTYPE type_x() const { return intype_x; }
        DIMTYPE type_y() const { return intype_y; }
        DIMTYPE type_z() const { return intype_z; }

        /* Viscosity, diffusivity, and Coriolis frequency */
        double get_visco() const { return VISCO; }
        double get_diffusivity(int t_num) const {
            switch (t_num) {
                case RHO:
                    return kappa_rho;
                case 1:
                    return kappa_trc;
                default:
                    assert(0 && "Invalid tracer number!");
            }
        }
        double get_rot_f() const { return rot_f; }

        /* Temporal values */
        double init_time() const { return initial_time; }
        int get_restart_sequence() const { return restart_sequence; }
        double get_plot_interval() const { return plot_interval; }
        double get_dt_max() const { return dt_max; }
        double get_next_plot() { return next_plot; }

        /* Number of tracers */
        int numActive() const { return 1; }
        int numPassive() const {
            if (tracer) return 1;
            else return 0;
        }

        /* Read grid */
        bool is_mapped() const { return mapped; }
        void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
            zgrid = alloc_array(Nx,Ny,Nz);

            if (input_data_types == MATLAB) {
                init_matlab("xgrid","x2d",xg);
                init_matlab("zgrid","z2d",zg);
            } else if (input_data_types == CTYPE || input_data_types == FULL3D) {
                init_ctype("xgrid","x2d",xg);
                init_ctype("zgrid","z2d",zg);
            }
            *zgrid = zg;
            // Automatically generate y-grid
            yg = 0*ii + MinY + Ly*(0.5+jj)/Ny + 0*kk;

            // Write the arrays and matlab readers
            write_array(xg,"xgrid");
            write_reader(xg,"xgrid",false);
            if (Ny > 1 || rot_f != 0) {
                write_array(yg,"ygrid");
                write_reader(yg,"ygrid",false);
            }
            write_array(zg,"zgrid");
            write_reader(zg,"zgrid",false);
        }

        /* Initialize velocities */
        void init_vels(DTArray & u, DTArray & v, DTArray & w) {
            if (master()) fprintf(stdout,"Initializing velocities\n");
            // if restarting
            if (restarting and (!restart_from_dump)) {
                init_vels_restart(u, v, w);
            }
            else if (restarting and restart_from_dump) {
                init_vels_dump(u, v, w);
            }
            // else start from other data formats
            else {
                switch(input_data_types) {
                    case MATLAB: // MATLAB data
                        init_vels_matlab(u, v, w, u_filename, v_filename, w_filename);
                        break;
                    case CTYPE: // Column-major 2D data
                        init_vels_ctype(u, v, w, u_filename, v_filename, w_filename);
                        break;
                    case FULL3D:
                        assert(0 && "If Full 3D fields already exist, turn restart on.");
                }

                // Add a random perturbation to trigger any 3D instabilities
                if ( !restarting ) {
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

                // Write the arrays and matlab readers
                write_reader(u,"u",true);
                write_reader(w,"w",true);
                write_array(u,"u",plotnum);
                write_array(w,"w",plotnum);
                if (Ny > 1 || rot_f != 0) {
                    write_array(v,"v",plotnum);
                    write_reader(v,"v",true);
                }
            }
        }

        /* Initialze the tracers (density, and dyes) */
        void init_tracer(int t_num, DTArray & tracer) {
            if (t_num == 0) {
                /* Density */
                // if restarting
                if (restarting and (!restart_from_dump)) {
                    init_tracer_restart("rho",tracer);
                }
                else if (restarting and restart_from_dump) {
                    init_tracer_dump("rho",tracer);
                }
                else {
                    // else start from other data formats
                    switch (input_data_types) {
                        case MATLAB:
                            init_matlab("rho",rho_filename,tracer);
                            break;
                        case CTYPE:
                            init_ctype("rho",rho_filename,tracer);
                            break;
                        case FULL3D:
                            assert(0 && "If Full 3D fields already exist, turn restart on.");
                    }
                    // Write the arrays and matlab readers
                    write_array(tracer,"rho",plotnum);
                    write_reader(tracer,"rho",true);
                    // and for pressure if it is wanted
                    if (pressure_write)
                        write_reader(tracer,"p",true);
                }
            } else if (t_num == 1) {
                /* Passive tracer */
                // if restarting
                if (restarting and (!restart_from_dump)) {
                    init_tracer_restart("tracer",tracer);
                }
                else if (restarting and restart_from_dump) {
                    init_tracer_dump("tracer",tracer);
                }
                else {
                    // else start from other data formats
                    switch (input_data_types) {
                        case MATLAB:
                            init_matlab("tracer",tracer_filename,tracer);
                        case CTYPE:
                            init_ctype("tracer",tracer_filename,tracer);
                        case FULL3D:
                            assert(0 && "If Full 3D fields already exist, turn restart on.");
                    }
                    // Write the arrays and matlab readers
                    write_array(tracer,"tracer",plotnum);
                    write_reader(tracer,"tracer",true);
                }
            }
        }

        /* Forcing in the momentum equations */
        void forcing(double t, DTArray & u, DTArray & u_f,
                DTArray & v, DTArray & v_f, DTArray & w, DTArray & w_f,
                vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
            u_f = +rot_f*v;
            v_f = -rot_f*u;
            w_f = -g*((*tracers[0]))/rho_0;
            *(tracers_f[0]) = 0;
            if (tracer) {
                *(tracers_f[1]) = 0;
                w_f = w_f - tracer_g*((*tracers[1]));
            }
        }

        /* Basic analysis, to write out the field periodically */
        void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
                vector<DTArray *> tracer, DTArray & pressure) {

            /* Write to disk if at correct time */
            if ((time - next_plot) > -1e-6*plot_interval) {
                plotnum++;
                t_step = MPI_Wtime(); // time just before write (for dump)
                // Write the arrays
                write_array(u,"u",plotnum);
                if (Ny > 1 || rot_f != 0)
                    write_array(v,"v",plotnum);
                write_array(w,"w",plotnum);
                write_array(*tracer[RHO],"rho",plotnum);
                if (pressure_write)
                    write_array(pressure,"p",plotnum);
                if (tracer.size() > 1)
                    write_array(*tracer[1],"tracer",plotnum);
                // update next plot time
                next_plot = next_plot + plot_interval;

                // Find average time to write (for dump)
                clock_time = MPI_Wtime(); // time just afer write
                avg_write_time = (avg_write_time*(plotnum-restart_sequence-1)
                        + (clock_time - t_step))/(plotnum-restart_sequence);
                // Print information about plot outputs
                write_plot_times(clock_time-t_step, avg_write_time, plot_interval,
                        plotnum, restarting, time);
            }
            // increase counter and update clocks
            itercount++;
            if (master()) {
                clock_time = MPI_Wtime();
                if (itercount == 1) {
                    step_start_time = MPI_Wtime(); // beginning of simulation (after reading in data)
                } else {
                    t_step = clock_time - step_start_time;
                }
            }

            // Also, calculate and write out useful information: maximum u, v, w...
            double max_u = psmax(max(abs(u)));
            double max_v = psmax(max(abs(v)));
            double max_w = psmax(max(abs(w)));
            double max_ke = psmax(max(0.5*rho_0*(u*u + v*v + w*w)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            double ke_tot = pssum(sum(0.5*rho_0*(u*u + v*v + w*w)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            double pe_tot = pssum(sum((rho_0*(1+*tracer[RHO]))*g*((*zgrid)(ii,jj,kk) - MinZ)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk))); // assumes density is density anomaly
            double max_rho = psmax(max(abs(*tracer[RHO])));
            if (master() and itercount == 1 and !restarting) {
                // create file for other analysis variables and write the column headers
                double t_startup = clock_time - real_start_time;
                fprintf(stdout,"Start-up time: %.6g s.\n",t_startup);
                FILE * analysis_file = fopen("analysis.txt","a");
                assert(analysis_file);
                fprintf(analysis_file,"Iter, Clock time, Sim time, "
                        "Max U, Max V, Max W, Max Density");
                if (tracer.size() > 1)
                    fprintf(analysis_file,", Max Tracer");
                fprintf(analysis_file,"\n");
                fclose(analysis_file);
            }
            if (master()) {
                /* add to the analysis file at each time step */
                FILE * analysis_file = fopen("analysis.txt","a");
                assert(analysis_file);
                fprintf(analysis_file,"%d %.12g %.12f "
                        "%.12g %.12g %.12g %.12g ",
                        itercount,t_step,time,
                        max_u,max_v,max_w,max_rho);
                if (tracer.size() > 1){
                    double max_dye = psmax(max(abs(*tracer[1])));
                    fprintf(analysis_file," %.12g",max_dye);
                }
                fprintf(analysis_file,"\n");
                fclose(analysis_file);
                /* and to the log file */
                fprintf(stdout,"[%d] (%.4g) %.4f: "
                        "%.4g %.4g %.4g "
                        "%.4g",
                        itercount,t_step,time,
                        max_u,max_v,max_w,max_rho);
                if (tracer.size() > 1){
                    double max_dye = psmax(max(abs(*tracer[1])));
                    fprintf(stdout," %.4g",max_dye);
                }
                fprintf(stdout,"\n");
            }

            // Determine last plot if restarting from the dump case
            if (restart_from_dump and (itercount == 1)){
                last_plot = restart_sequence*plot_interval;    
                next_plot = last_plot + plot_interval;
            }
            // see if close to end of compute time and dump
            check_and_dump(clock_time, real_start_time, compute_time, time, avg_write_time,
                    plotnum, u, v, w, tracer);
            /* Change dump log file if successfully reached final time
               the dump time will be twice final time so that a restart won't actually start */
            successful_dump(plotnum, final_time, plot_interval);
        }

        // User specified variables to dump
        void write_variables(DTArray & u,DTArray & v, DTArray & w,
                vector<DTArray *> & tracer) {
            write_array(u,"u.dump",-1);
            write_array(v,"v.dump",-1);
            write_array(w,"w.dump",-1);
            write_array(*tracer[RHO],"rho.dump",-1);
            if (tracer.size() > 1)
                write_array(*tracer[1],"tracer.dump",-1);
        }

        userControl() :
            // Initialize the local variables
            plotnum(restart_sequence), 
            next_plot(restart_time + plot_interval)
    {   compute_quadweights(
            size_x(),   size_y(),   size_z(),
            length_x(), length_y(), length_z(),
            type_x(),   type_y(),   type_z());
    // If this is an unmapped grid, generate/write the
    // 3D grid files
    if (!is_mapped()) {
        automatic_grid(MinX, MinY, MinZ);
    }
    }
};

/* The ''main'' routine */
int main(int argc, char ** argv) {
    /* Initialize MPI.  This is required even for single-processor runs,
       since the inner routines assume some degree of parallelization,
       even if it is trivial. */
    MPI_Init(&argc,&argv);
    /* Change filtering from default if you want to */
    //f_strength = -.25;
    //f_cutoff = 0.6;
    //f_order = 4;

    real_start_time = MPI_Wtime();     // start of simulation (for dump)
    /* ------------------ Define parameters from spins.conf --------------------- */
    options_init();

    option_category("Grid Options");
    add_option("Lx",&Lx,"X-length");
    add_option("Ly",&Ly,1.0,"Y-length");
    add_option("Lz",&Lz,"Z-length");
    add_option("Nx",&Nx,"Number of points in X");
    add_option("Ny",&Ny,1,"Number of points in Y");
    add_option("Nz",&Nz,"Number of points in Z");
    add_option("min_x",&MinX,0.0,"Unmapped grids: Minimum X-value");
    add_option("min_y",&MinY,0.0,"Minimum Y-value");
    add_option("min_z",&MinZ,0.0,"Minimum Z-value");

    string xgrid_type, ygrid_type, zgrid_type;
    add_option("type_x",&xgrid_type,
            "Grid type in X.  Valid values are:\n"
            "   FOURIER: Periodic\n"
            "   FREE_SLIP: Cosine expansion\n"
            "   NO_SLIP: Chebyhsev expansion");
    add_option("type_y",&ygrid_type,"FOURIER","Grid type in Y");
    add_option("type_z",&zgrid_type,"Grid type in Z");

    option_category("Grid mapping options");
    add_option("mapped_grid",&mapped,false,"Is the grid mapped?");
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

    add_option("u_file",&u_filename,"U-velocity filename");
    add_option("v_file",&v_filename,"","V-velocity filename");
    add_option("w_file",&w_filename,"W-velocity filename");
    add_option("rho_file",&rho_filename,"Rho (density) filename");
    add_option("tracer_file",&tracer_filename,"Tracer filename");

    option_category("Physical parameters");
    add_option("g",&g,9.81,"Gravitational acceleration");
    add_option("rot_f",&rot_f,0.0,"Coriolis force term");
    add_option("rho_0",&rho_0,1.0,"Reference density");
    add_option("visco",&VISCO,0.0,"Kinematic viscosity");
    add_option("kappa",&kappa_rho,0.0,"Thermal diffusivity");

    option_category("Second tracer");
    add_switch("enable_tracer",&tracer,"Enable evolution of a second tracer");
    add_option("tracer_kappa",&kappa_trc,"Diffusivity of tracer");
    add_option("tracer_gravity",&tracer_g,9.81,"Gravity for the second tracer");

    option_category("Running options");
    add_option("initial_time",&initial_time,0.0,"Initial time");
    add_option("final_time",&final_time,"Final time");
    add_option("plot_interval",&plot_interval,"Interval between output times");
    add_option("dt_max",&dt_max,0.1,"Maximum time step");

    option_category("Restart options");
    add_option("restart",&restarting,false,"Restart from prior output time.  OVERRIDES many other values.");
    add_option("restart_time",&restart_time,0.0,"Time to restart from");
    add_option("restart_sequence",&restart_sequence,-1,"Sequence number to restart from");

    option_category("Dumping options");
    add_option("restart_from_dump",&restart_from_dump,false,"If restart from dump");
    add_option("compute_time",&compute_time,-1.0,"Time permitted for computation");

    option_category("Write pressure");
    add_option("pressure_write",&pressure_write,false,"Enable the outputting of the pressure");
    add_option("perturb",&perturb,0.0,"Velocity perturbation (multiplicative white noise) applied to read-in data.");

    // Parse the options from the command line and config file
    options_parse(argc,argv);

    /* Now, make sense of the options received.  Many of these values
       can be directly used, but the ones of string-type need further
       procesing. */

    /* ------------------ Adjust for starting from a dump --------------------- */
    // Read information from dump_time.txt
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

        // Kill simulation if already past final time
        if (restart_time > final_time){
            if (master()){
                fprintf(stderr,"Restart dump time (%.4g) is past final time (%.4g). Quitting now.\n",restart_time,final_time);
            }
            MPI_Finalize(); exit(1);
        }
    }
    // Estimate the time to write
    if (compute_time > 0){
        avg_write_time = max(100.0*Nx*Ny*Nz/pow(512.0,3), 20.0);
    }

    /* ------------------ Set grid and file types --------------------- */
    // x
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
    // y
    if (ygrid_type == "FOURIER") {
        intype_y = PERIODIC;
    } else if (ygrid_type == "FREE_SLIP") {
        intype_y = FREE_SLIP;
    } else {
        if (master())
            fprintf(stderr,"Invalid option %s received for type_y\n",ygrid_type.c_str());
        MPI_Finalize(); exit(1);
    }
    // z
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
                fprintf(stderr,"Warning: the initial time (%g) does not appear to be an even multiple of the plot interval (%g)\n",
                        initial_time,plot_interval);
            }
        }
    }

    /* ------------------ Do stuff --------------------- */
    userControl mycode; // Create an instantiated object of the above class
    // Create a flow-evolver that takes its settings from the above class
    FluidEvolve<userControl> kevin_kh(&mycode);
    // Initialize
    kevin_kh.initialize();
    // Run until the end of time
    kevin_kh.do_run(final_time);
    MPI_Finalize(); // Cleanly exit MPI
    return 0; // End the program
}
