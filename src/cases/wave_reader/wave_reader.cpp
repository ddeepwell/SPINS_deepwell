/* wave_reader.cpp -- general case for looking at the evolution of
   waves, with input data and configuration provided at runtime
   via a configuration file. */

/* ------------------ Top matter --------------------- */

// Required headers
#include "../BaseCase.hpp"      // contains default class
#include "../Options.hpp"       // config-file parser
#include <random/normal.h>      // Blitz random number generator

using ranlib::Normal;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

/* ------------------ Define parameters --------------------- */

// Grid scales
double Lx, Ly, Lz;          // Grid lengths (m)
int    Nx, Ny, Nz;          // Number of points in x, y, z
double MinX, MinY, MinZ;    // Minimum x/y/z points
// Mapped grid?
bool mapped;
// Grid types
DIMTYPE intype_x, intype_y, intype_z;
string grid_type[3];

// Physical parameters
double g, rot_f, rho_0;     // gravity accel (m/s^2), Coriolis frequency (s^-1), reference density (kg/L)
double visco;               // viscosity (m^2/s)
double mu;                  // dynamic viscosity (kg/(m·s))
double kappa_rho;           // diffusivity of density (m^2/s)

// tracer options
const int RHO = 0;          // index for rho
const int TRCR = 1;         // index for tracer
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
double avg_write_time;
double startup_time;
double step_start_time;

// other options
double perturb;             // Initial velocity perturbation
bool write_pressure;        // Write out pressure?
bool compute_stress;        // Compute surface stresses?
bool compute_enstrophy;     // Compute Enstrophy?
bool compute_dissipation;   // Compute dissipation?
int itercount = 0;          // Iteration counter

// Input file names
string xgrid_filename,
       ygrid_filename,
       zgrid_filename,
       u_filename,
       v_filename,
       w_filename,
       rho_filename,
       tracer_filename;

/* ------------------ Adjust the class --------------------- */

class userControl : public BaseCase {
    public:
        /* Grid arrays */
        DTArray *zgrid;
        Array<double,1> xx, yy, zz;

        /* arrays and operators for derivatives */
        Grad * gradient_op;
        DTArray *temp1;
        DTArray *Hprime;

        /* Timing variables (for outputs and measuring time steps) */
        int plotnum;        // most recent output number (for plotting)
        double last_plot;   // most recent output time
        double next_plot;   // time of next output write
        // variables for timing steps
        double t_step;
        double clock_time;

        /* Size of domain */
        double length_x() const { return Lx; }
        double length_y() const { return Ly; }
        double length_z() const { return Lz; }

        /* Resolution in X, Y, and Z */
        int size_x() const { return Nx; }
        int size_y() const { return Ny; }
        int size_z() const { return Nz; }

        /* Set expansions (FREE_SLIP, NO_SLIP (in vertical) or PERIODIC) */
        DIMTYPE type_x() const { return intype_x; }
        DIMTYPE type_y() const { return intype_y; }
        DIMTYPE type_z() const { return intype_z; }

        /* Record the gradient-taking object */
        void set_grad(Grad * in_grad) { gradient_op = in_grad; }

        /* Viscosity, diffusivity, and Coriolis frequency */
        double get_rot_f() const { return rot_f; }
        double get_visco() const { return visco; }
        double get_diffusivity(int t_num) const {
            switch (t_num) {
                case RHO:
                    return kappa_rho;
                case TRCR:
                    return kappa_trc;
                default:
                    if (master()) fprintf(stderr,"Invalid tracer number!\n");
                    MPI_Finalize(); exit(1);
            }
        }

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
            if (restarting and !restart_from_dump) {
                init_vels_restart(u, v, w);
            } else if (restarting and restart_from_dump) {
                init_vels_dump(u, v, w);
            } else {
                // else start from other data formats
                switch(input_data_types) {
                    case MATLAB: // MATLAB data
                        init_vels_matlab(u, v, w, u_filename, v_filename, w_filename);
                        break;
                    case CTYPE: // Column-major 2D data
                        init_vels_ctype(u, v, w, u_filename, v_filename, w_filename);
                        break;
                    case FULL3D:
                        if (master()) fprintf(stderr,"FULL3D option chosen, turn restart on\n");
                        MPI_Finalize(); exit(1);
                }

                // Add a random perturbation to trigger any 3D instabilities
                int myrank;
                MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
                Normal<double> rnd(0,1);
                for (int i = u.lbound(firstDim); i <= u.ubound(firstDim); i++) {
                    rnd.seed(i);
                    for (int j = u.lbound(secondDim); j <= u.ubound(secondDim); j++) {
                        for (int k = u.lbound(thirdDim); k <= u.ubound(thirdDim); k++) {
                            u(i,j,k) *= 1+perturb*rnd.random();
                            if ( Ny > 1 )
                                v(i,j,k) *= 1+perturb*rnd.random();
                            w(i,j,k) *= 1+perturb*rnd.random();
                        }
                    }
                }

                // Write the arrays
                write_array(u,"u",plotnum);
                write_array(w,"w",plotnum);
                if (Ny > 1 || rot_f != 0) {
                    write_array(v,"v",plotnum);
                }
            }
        }

        /* Initialize the tracers (density and dyes) */
        void init_tracers(vector<DTArray *> & tracers) {
            if (master()) fprintf(stdout,"Initializing tracers\n"); 
            // Sanity checks
            assert(numtracers() == int(tracers.size()));
            assert(numtracers() >= 1);

            // if restarting
            if (restarting and !restart_from_dump) {
                init_tracer_restart("rho",*tracers[RHO]);
                if (tracer)
                    init_tracer_restart("tracer",*tracers[TRCR]);
            } else if (restarting and restart_from_dump) {
                init_tracer_dump("rho",*tracers[RHO]);
                if (tracer)
                    init_tracer_dump("tracer",*tracers[TRCR]);
            } else {
                // else start from other data formats
                switch (input_data_types) {
                    case MATLAB:
                        init_matlab("rho",rho_filename,*tracers[RHO]);
                        if (tracer)
                            init_matlab("tracer",tracer_filename,*tracers[TRCR]);
                        break;
                    case CTYPE:
                        init_ctype("rho",rho_filename,*tracers[RHO]);
                        if (tracer)
                            init_ctype("tracer",tracer_filename,*tracers[TRCR]);
                        break;
                    case FULL3D:
                        if (master()) fprintf(stderr,"FULL3D option chosen, turn restart on\n");
                        MPI_Finalize(); exit(1);
                }
                // Write the arrays
                write_array(*tracers[RHO],"rho",plotnum);
                if (tracer)
                    write_array(*tracers[TRCR],"tracer",plotnum);
            }
        }

        /* Forcing in the momentum equations */
        void forcing(double t, DTArray & u, DTArray & u_f,
                DTArray & v, DTArray & v_f, DTArray & w, DTArray & w_f,
                vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
            u_f = +rot_f*v;
            v_f = -rot_f*u;
            w_f = -g*(*tracers[RHO]);   // tracer[RHO] = rho/rho_0
            *tracers_f[RHO] = 0;
            if (tracer) {
                *tracers_f[TRCR] = 0;
                w_f = w_f - tracer_g*(*tracers[TRCR]);
            }
        }

        /* Basic analysis, to write out the field periodically */
        void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
                vector<DTArray *> tracers, DTArray & pressure) {
            // increase counter
            itercount++;
            // Set-up
            if ( itercount == 1 ) {
                temp1 = alloc_array(Nx,Ny,Nz);
                if (compute_stress) {
                    Hprime = alloc_array(Nx,Ny,1);
                    if (mapped) {
                        bottom_slope(*Hprime, *zgrid, *temp1, gradient_op, grid_type, Nx, Ny, Nz);
                    } else {
                        *Hprime = 0;
                    }   
                }
            }

            /* Write to disk if at correct time */
            if ((time - next_plot) > -1e-6*plot_interval) {
                plotnum++;
                t_step = MPI_Wtime(); // time just before write (for dump)
                // Write the arrays
                write_array(u,"u",plotnum);
                if (Ny > 1 || rot_f != 0)
                    write_array(v,"v",plotnum);
                write_array(w,"w",plotnum);
                write_array(*tracers[RHO],"rho",plotnum);
                if (tracer)
                    write_array(*tracers[TRCR],"tracer",plotnum);
                if (write_pressure)
                    write_array(pressure,"p",plotnum);
                // update next plot time
                next_plot = next_plot + plot_interval;

                // Find average time to write (for dump)
                clock_time = MPI_Wtime(); // time just after write
                avg_write_time = (avg_write_time*(plotnum-restart_sequence-1)
                        + (clock_time - t_step))/(plotnum-restart_sequence);
                // Print information about plot outputs
                write_plot_times(clock_time-t_step, avg_write_time, plot_interval,
                        plotnum, restarting, time);
            }
            // update clocks
            if (master()) {
                clock_time = MPI_Wtime();
                t_step = clock_time - step_start_time;
            }

            // Also, calculate and write out useful information

            // total dissipation
            double diss_tot = 0;
            if (compute_dissipation) {
                dissipation(u, v, w, *temp1, gradient_op, grid_type, Nx, Ny, Nz, mu);
                diss_tot = pssum(sum((*temp1)*
                            (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            }
            // Energy (PE assumes density is density anomaly)
            double ke_tot = pssum(sum(0.5*rho_0*(u*u + v*v + w*w)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            double pe_tot;
            if (mapped) {
                pe_tot = pssum(sum(rho_0(1+*tracers[RHO])*g*((*zgrid)(ii,jj,kk) - MinZ)*
                            (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            } else {
                pe_tot = pssum(sum(rho_0(1+*tracers[RHO])*g*(zz(kk) - MinZ)*
                            (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            }
            // max of fields
            double max_u = psmax(max(abs(u)));
            double max_v = psmax(max(abs(v)));
            double max_w = psmax(max(abs(w)));
            double max_ke = psmax(max(0.5*rho_0*(u*u + v*v + w*w)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            double max_rho = psmax(max(abs(*tracers[RHO])));
            if (master() and itercount == 1 and !restarting) {
                // create file for other analysis variables and write the column headers
                FILE * analysis_file = fopen("analysis.txt","a");
                assert(analysis_file);
                fprintf(analysis_file,"Iter, Clock_time, Sim_time, "
                        "Max_U, Max_V, Max_W, "
                        "Max_KE, Total_KE, Total_PE, Total_dissipation, "
                        "Max_density");
                if (tracer)
                    fprintf(analysis_file,", Max_tracer");
                fprintf(analysis_file,"\n");
                fclose(analysis_file);
            }
            if (master()) {
                /* add to the analysis file at each time step */
                FILE * analysis_file = fopen("analysis.txt","a");
                assert(analysis_file);
                fprintf(analysis_file,"%d, %.12g, %.12f, "
                        "%.12g, %.12g, %.12g, "
                        "%.12g, %.12g, %.12g, %.12g, "
                        "%.12g",
                        itercount,t_step,time,
                        max_u,max_v,max_w,
                        max_ke,ke_tot,pe_tot,diss_tot,
                        max_rho);
                if (tracer){
                    double max_dye = psmax(max(abs(*tracers[TRCR])));
                    fprintf(analysis_file,", %.12g",max_dye);
                }
                fprintf(analysis_file,"\n");
                fclose(analysis_file);
                /* and to the log file */
                fprintf(stdout,"[%d] (%.4g) %.4f: "
                        "%.4g %.4g %.4g "
                        "%.4g %.4g %.4g %.4g "
                        "%.4g",
                        itercount,t_step,time,
                        max_u,max_v,max_w,
                        max_ke,ke_tot,pe_tot,diss_tot,
                        max_rho);
                if (tracer){
                    double max_dye = psmax(max(abs(*tracers[TRCR])));
                    fprintf(stdout," %.4g",max_dye);
                }
                fprintf(stdout,"\n");
            }

            // compute other things, if wanted
            if (compute_stress) {
                stresses(u, v, w, *Hprime, *temp1, gradient_op, grid_type,
                        mu, time, itercount, restarting);
            }
            if (compute_enstrophy) {
                enstrophy(u, v, w, *temp1, gradient_op, grid_type,
                        time, itercount, restarting);
            }

            // Determine last plot if restarting from the dump case
            if (restart_from_dump and itercount == 1) {
                last_plot = restart_sequence*plot_interval;    
                next_plot = last_plot + plot_interval;
            }
            // see if close to end of compute time and dump
            check_and_dump(clock_time, real_start_time, compute_time, time, avg_write_time,
                    plotnum, itercount, u, v, w, tracers);
            // Change dump log file if successfully reached final time
            successful_dump(plotnum, final_time, plot_interval);
        }

        // User specified variables to dump
        void write_variables(DTArray & u,DTArray & v, DTArray & w,
                vector<DTArray *> & tracers) {
            write_array(u,"u.dump",-1);
            write_array(v,"v.dump",-1);
            write_array(w,"w.dump",-1);
            write_array(*tracers[RHO],"rho.dump",-1);
            if (tracer)
                write_array(*tracers[TRCR],"tracer.dump",-1);
        }

        // Constructor: Initialize local variables
        userControl() :
            xx(split_range(Nx)), yy(Ny), zz(Nz),
            gradient_op(0),
            plotnum(restart_sequence),
            next_plot(restart_time + plot_interval)
    {   compute_quadweights(
            size_x(),   size_y(),   size_z(),
            length_x(), length_y(), length_z(),
            type_x(),   type_y(),   type_z());
    // If this is an unmapped grid, generate/write the
    // 3D grid files
    if (!is_mapped()) {
        automatic_grid(MinX, MinY, MinZ, &xx, &yy, &zz);
    }
    }
};

/* The ''main'' routine */
int main(int argc, char ** argv) {
    /* Initialize MPI.  This is required even for single-processor runs,
       since the inner routines assume some degree of parallelization,
       even if it is trivial. */
    MPI_Init(&argc, &argv);

    real_start_time = MPI_Wtime();     // start of simulation (for dump)
    /* ------------------ Define parameters from spins.conf --------------------- */
    options_init();

    option_category("Grid Options");
    add_option("Lx",&Lx,"Length of tank");
    add_option("Ly",&Ly,1.0,"Width of tank");
    add_option("Lz",&Lz,"Height of tank");
    add_option("Nx",&Nx,"Number of points in X");
    add_option("Ny",&Ny,1,"Number of points in Y");
    add_option("Nz",&Nz,"Number of points in Z");
    add_option("min_x",&MinX,0.0,"Minimum X-value");
    add_option("min_y",&MinY,0.0,"Minimum Y-value");
    add_option("min_z",&MinZ,0.0,"Minimum Z-value");

    option_category("Grid expansion options");
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
    add_option("rot_f",&rot_f,0.0,"Coriolis frequency");
    add_option("rho_0",&rho_0,1000.0,"Reference density");
    add_option("visco",&visco,0.0,"Kinematic viscosity");
    add_option("kappa_rho",&kappa_rho,0.0,"Diffusivity of density");

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

    option_category("Other options");
    add_option("write_pressure",&write_pressure,false,"Enable the outputting of the pressure");
    add_option("perturb",&perturb,0.0,"Velocity perturbation applied to read-in data.");
    add_option("compute_stress",&compute_stress,true,"Calculate the top and bottom stresses?");
    add_option("compute_enstrophy",&compute_enstrophy,true,"Calculate enstrophy?");
    add_option("compute_dissipation",&compute_dissipation,true,"Calculate dissipation?");

    option_category("Filter options");
    add_option("f_cutoff",&f_cutoff,0.6,"Filter cut-off frequency");
    add_option("f_order",&f_order,2.0,"Filter order");
    add_option("f_strength",&f_strength,20.0,"Filter strength");

    // Parse the options from the command line and config file
    options_parse(argc,argv);

    /* ------------------ Adjust and check parameters --------------------- */
    /* Now, make sense of the options received.  Many of these values
       can be directly used, but the ones of string-type need further procesing. */

    // adjust time if starting from a dump
    if (restart_from_dump) {
        adjust_for_dump(restarting, restart_time, restart_sequence,
                final_time, compute_time, avg_write_time, Nx, Ny, Nz);
    }

    // check the restart sequence
    check_restart_sequence(restarting, restart_sequence, initial_time, restart_time, plot_interval);

    // parse file types
    get_datatype(datatype, input_data_types);
    // parse expansion types
    get_boundary_conditions(xgrid_type, ygrid_type, zgrid_type, intype_x, intype_y, intype_z);
    // vector of string types
    grid_type[0] = xgrid_type;
    grid_type[1] = ygrid_type;
    grid_type[2] = zgrid_type;

    // adjust Ly for 2D
    if (Ny==1 and Ly!=1.0){
        Ly = 1.0;
        if (master())
            fprintf(stdout,"Simulation is 2 dimensional, "
                    "Ly has been changed to 1.0 for normalization.\n");
    }

    /* ------------------ Derived parameters --------------------- */

    // Dynamic viscosity
    mu = visco*rho_0;

    /* ------------------ Print some parameters --------------------- */
    if (master()) {
        fprintf(stdout,"Wave reader problem\n");
        fprintf(stdout,"Using a %f x %f x %f grid of %d x %d x %d points\n",Lx,Ly,Lz,Nx,Ny,Nz);
        fprintf(stdout,"g = %f, rot_f = %f, rho_0 = %f\n",g,rot_f,rho_0);
        fprintf(stdout,"Time between plots: %g s\n",plot_interval);
        fprintf(stdout,"Initial velocity perturbation: %g\n",perturb);
        fprintf(stdout,"Filter cutoff = %f, order = %f, strength = %f\n",f_cutoff,f_order,f_strength);
    }

    /* ------------------ Do stuff --------------------- */
    userControl mycode; // Create an instance of the above class
    FluidEvolve<userControl> kevin_kh(&mycode);
    kevin_kh.initialize();
    step_start_time = MPI_Wtime(); // beginning of simulation (after reading in data)
    startup_time = step_start_time - real_start_time;
    if (master()) fprintf(stdout,"Start-up time: %.6g s.\n",startup_time);
    // Run until the end of time
    kevin_kh.do_run(final_time);
    MPI_Finalize();
    return 0;
}
