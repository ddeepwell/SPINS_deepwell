/* Generic script for two layer fluid with zero initial velocity
   and no topography */

/* ------------------ Top matter --------------------- */

// Required headers
#include "../BaseCase.hpp"      // Support file that contains default implementations of many functions
#include "../Options.hpp"       // config-file parser
#include <random/normal.h>      // Blitz random number generator
#include <fstream>

using namespace ranlib;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

/* ------------------ Parameters --------------------- */

// Grid scales
double Lx, Ly, Lz;          // Grid lengths (m)
int    Nx, Ny, Nz;          // Number of points in x, y (span), z
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
// helpful constants
const int NUM_TRACER = 1;
const int RHO = 0;          // index for rho

// Stratification parameters
double delta_rho;           // change in density (kg/L)
// pycnocline location parameters
double pyc_asym;            // shift of pycnocline above the mid-depth (m)
double pyc_sep;             // total separation of double pycnocline (m)
double h_halfwidth;         // pycnocline half-width (m)
double h_mix_half;          // vertical half-width transition of mixed region (m)
// Horizontal stratification parameters
double delta_x;             // horizontal transition length (m)
double Lmix;                // Width of mixed region (m)
double Hmix;                // Total height of mixed region (m)

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
double t_startup;
double step_start_time;

// other options
bool write_pressure;        // Write out pressure?
double perturb;             // Initial velocity perturbation
bool compute_stress;        // Compute surface stresses?
bool compute_enstrophy;     // Compute Enstrophy?
bool compute_dissipation;   // Compute dissipation?
int itercount = 0;          // Iteration counter

/* ------------------ Derived parameters --------------------- */

// Flow speed
double c0;

// Squared maximum buoyancy frequency if the initial stratification was stable
double N2_max;

// Reynolds number
double Re;

/* ------------------ Initialize the class --------------------- */

class dambreak : public BaseCase {
    public:
        /* Grid arrays */
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

        // Record the gradient-taking object.  This is given by the NSIntegrator
        // code, and it reflects the boundary types and any Jacobian-transform
        void set_grad(Grad * in_grad) { gradient_op = in_grad; }

        /* Viscosity, diffusivity, and Coriolis frequency */
        double get_rot_f() const { return rot_f; }
        double get_visco() const { return visco; }
        double get_diffusivity(int t_num) const {
            switch (t_num) {
                case RHO:
                    return kappa_rho;
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
        int numActive() const { return NUM_TRACER; }

        /* Create mapped grid */
        bool is_mapped() const { return mapped; }

        /* Initialize velocities */
        void init_vels(DTArray & u, DTArray & v, DTArray & w) {
            // if restarting
            if (restarting and (!restart_from_dump)) {
                init_vels_restart(u, v, w);
            } else if (restarting and restart_from_dump) {
                init_vels_dump(u, v, w);
            } else{
                // else have a near motionless field
                u = 0;
                v = 0;
                w = 0;
                // Add a random perturbation to trigger any 3D instabilities
                int myrank;
                MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
                Normal<double> rnd(0,1);
                for (int i = u.lbound(firstDim); i <= u.ubound(firstDim); i++) {
                    rnd.seed(i);
                    for (int j = u.lbound(secondDim); j <= u.ubound(secondDim); j++) {
                        for (int k = u.lbound(thirdDim); k <= u.ubound(thirdDim); k++) {
                            u(i,j,k) += perturb*rnd.random();
                            if ( Ny > 1)
                                v(i,j,k) += perturb*rnd.random();
                            w(i,j,k) += perturb*rnd.random();
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

        /* Initialze the tracers (density and dyes) */
        void init_tracers(vector<DTArray *> & tracers) {
            if (master()) fprintf(stdout,"Initializing Density\n");
            DTArray & rho = *tracers[RHO];
            assert(numtracers() == int(tracers.size()));

            if (restarting and (!restart_from_dump)) {
                init_tracer_restart("rho",rho);
            } else if (restarting and restart_from_dump) {
                init_tracer_dump("rho",rho);
            } else {
                // background stratification
                rho =  -0.25*delta_rho*tanh((zz(kk)-(MinZ+0.5*Lz+pyc_asym-0.5*pyc_sep))/h_halfwidth);
                rho += -0.25*delta_rho*tanh((zz(kk)-(MinZ+0.5*Lz+pyc_asym+0.5*pyc_sep))/h_halfwidth);
                rho = rho*0.5*(1.0+tanh((xx(ii)-Lmix)/delta_x));
                // mixed region
                rho = rho + 0.5*(1.0-tanh((xx(ii)-Lmix)/delta_x))
                    *(-0.25*delta_rho)*(
                            1.0+tanh((zz(kk)-(MinZ+0.5*Lz+pyc_asym)+0.5*Hmix)/h_mix_half)
                            -1.0+tanh((zz(kk)-(MinZ+0.5*Lz+pyc_asym)-0.5*Hmix)/h_mix_half));
                // Write the arrays
                write_array(rho,"rho",plotnum); 
            }
        }

        /* Forcing in the momentum equations */
        void forcing(double t, DTArray & u, DTArray & u_f,
                DTArray & v, DTArray & v_f, DTArray & w, DTArray & w_f,
                vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
            u_f = +rot_f*v;
            v_f = -rot_f*u;
            w_f = -g*(*tracers[RHO])/rho_0;
            *tracers_f[RHO] = 0;
        }

        /* Basic analysis, to write out the field periodically */
        void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
                vector<DTArray *> & tracers, DTArray & pressure) {
            // increase counter
            itercount++;
            // Set-up
            if ( itercount == 1 ) {
                temp1 = alloc_array(Nx,Ny,Nz);
                if (compute_stress) {
                    Hprime = alloc_array(Nx,Ny,1);
                    Hprime = 0;
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
            double pe_tot = pssum(sum((rho_0*(1+*tracers[RHO]))*g*(zz(kk) - MinZ)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
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
                        "Max_density\n");
                fclose(analysis_file);
            }
            if (master()) {
                /* add to the analysis file at each time step */
                FILE * analysis_file = fopen("analysis.txt","a");
                assert(analysis_file);
                fprintf(analysis_file,"%d, %.12g, %.12f, "
                        "%.12g, %.12g, %.12g, "
                        "%.12g, %.12g, %.12g, %.12g, "
                        "%.12g\n",
                        itercount,t_step,time,
                        max_u,max_v,max_w,
                        max_ke,ke_tot,pe_tot,diss_tot,
                        max_rho);
                fclose(analysis_file);
                /* and to the log file */
                fprintf(stdout,"[%d] (%.4g) %.4f: "
                        "%.4g %.4g %.4g "
                        "%.4g %.4g %.4g %.4g "
                        "%.4g\n",
                        itercount,t_step,time,
                        max_u,max_v,max_w,
                        max_ke,ke_tot,pe_tot,diss_tot,
                        max_rho);
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
            if (restart_from_dump and (itercount == 1)){
                last_plot = restart_sequence*plot_interval;    
                next_plot = last_plot + plot_interval;
            }
            // see if close to end of compute time and dump
            check_and_dump(clock_time, real_start_time, compute_time, time, avg_write_time,
                    plotnum, itercount, u, v, w, tracers);
            /* Change dump log file if successfully reached final time
               the dump time will be twice final time so that a restart won't actually start */
            successful_dump(plotnum, final_time, plot_interval);
        }

        // User specified variables to dump
        void write_variables(DTArray & u,DTArray & v, DTArray & w,
                vector<DTArray *> & tracers) {
            write_array(u,"u.dump",-1);
            write_array(v,"v.dump",-1);
            write_array(w,"w.dump",-1);
            write_array(*tracers[RHO],"rho.dump",-1);
        }

        // Constructor: Initialize local variables
        dambreak():
            xx(split_range(Nx)), yy(Ny), zz(Nz),
            gradient_op(0),
            plotnum(restart_sequence),
            next_plot(restart_time + plot_interval)
    {   compute_quadweights(
            size_x(),   size_y(),   size_z(),
            length_x(), length_y(), length_z(),
            type_x(),   type_y(),   type_z());
    // Create one-dimensional arrays for the coordinates
    automatic_grid(MinX, MinY, MinZ, &xx, &yy, &zz);
    }
};

/* The ''main'' routine */
int main(int argc, char ** argv) {
    /* Initialize MPI.  This is required even for single-processor runs,
       since the inner routines assume some degree of parallelization,
       even if it is trivial. */
    MPI_Init(&argc, &argv);
    /* Change filtering from default if you want to */
    //f_strength = -0.33;
    //f_cutoff = 0.8;
    //f_order = 4;

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
    add_option("mapped_grid",&mapped,false,"Is the grid mapped?");

    option_category("Physical parameters");
    add_option("g",&g,9.81,"Gravitational acceleration");
    add_option("rot_f",&rot_f,0.0,"Coriolis frequency");
    add_option("rho_0",&rho_0,1.0,"Reference density");
    add_option("visco",&visco,0.0,"Viscosity");
    add_option("kappa_rho",&kappa_rho,0.0,"Diffusivity of density");	

    option_category("Stratification parameters");
    add_option("delta_rho",&delta_rho,"density difference");
    add_option("pyc_asym",&pyc_asym,"pycnocline vertical shift from mid-depth");
    add_option("pyc_sep",&pyc_sep,"total separation of double pycnocline");
    add_option("h_halfwidth",&h_halfwidth,"Pycnocline half-width");
    add_option("h_mix_half",&h_mix_half,"Pycnocline half-width");
    add_option("delta_x",&delta_x,"Horizontal transition half-width");
    add_option("Lmix",&Lmix,"Width of mixed region");
    add_option("Hmix",&Hmix,"Height of mixed region");

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
    add_option("perturb",&perturb,0.0,"Initial perturbation in velocity");
    add_option("compute_stress",&compute_stress,false,"Calculate the top and bottom stresses?");
    add_option("compute_enstrophy",&compute_enstrophy,true,"Calculate enstrophy?");
    add_option("compute_dissipation",&compute_dissipation,true,"Calculate dissipation?");

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
                fprintf(stderr,"Restart dump time (%.4g) is past final time (%.4g). "
                        "Quitting now.\n",restart_time,final_time);
            }
            MPI_Finalize(); exit(1);
        }
    }
    // Estimate the time to write
    if (compute_time > 0){
        avg_write_time = max(100.0*Nx*Ny*Nz/pow(512.0,3), 20.0);
    }

    /* ------------------ Set boundary conditions --------------------- */
    // x
    if (xgrid_type == "FOURIER") { intype_x = PERIODIC; }
    else if (xgrid_type == "FREE_SLIP") { intype_x = FREE_SLIP; }
    else if (xgrid_type == "NO_SLIP") { intype_x = NO_SLIP; }
    else {
        if (master())
            fprintf(stderr,"Invalid option %s received for type_x\n",xgrid_type.c_str());
        MPI_Finalize(); exit(1);
    }
    // y
    if (ygrid_type == "FOURIER" ) { intype_y = PERIODIC; }
    else if (ygrid_type == "FREE_SLIP") { intype_y = FREE_SLIP; }
    else {
        if (master())
            fprintf(stderr,"Invalid option %s received for type_y\n",ygrid_type.c_str());
        MPI_Finalize(); exit(1);
    }
    // z
    if (zgrid_type == "FOURIER") { intype_z = PERIODIC; }
    else if (zgrid_type == "FREE_SLIP") { intype_z = FREE_SLIP; }
    else if (zgrid_type == "NO_SLIP") { intype_z = NO_SLIP; }
    else {
        if (master())
            fprintf(stderr,"Invalid option %s received for type_z\n",zgrid_type.c_str());
        MPI_Finalize(); exit(1);
    }
    // check for proper expansion types
    if (mapped) {
        if (master())
            fprintf(stderr,"This case file is for unmapped cases only. "
                   "Change spins.conf, or use a different case file.\n");
        MPI_Finalize(); exit(1);
    }

    // vector of string types
    grid_type[0] = xgrid_type;
    grid_type[1] = ygrid_type;
    grid_type[2] = zgrid_type;

    /* ------------------ Adjust some parameters --------------------- */
    // adjust Ly for 2D
    if (Ny==1 and Ly!=1.0){
        Ly = 1.0;
        if (master()) {
            fprintf(stdout,"WARNING:\n");
            fprintf(stdout,"\tSimulation is 2 dimensional, Ly has been changed to 1.0 for normalization.\n");
        }
    }

    /* ------------------ Derived parameters --------------------- */

    // Dynamic viscosity
    mu = visco*rho_0;

    // Mode-2 wave speed
    c0 = 0.5*sqrt(g*h_halfwidth*delta_rho/rho_0);

    // Maximum buoyancy frequency (squared) if the initial stratification was stable
    N2_max = g/rho_0*delta_rho/(2*h_halfwidth);

    // Reynolds number
    Re = c0*h_halfwidth/visco;

    // Maximum time step
    dt_max = 0.5/sqrt(N2_max); 

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
                fprintf(stderr,"Warning: the initial time (%g) "
                        "does not appear to be an even multiple of the plot interval (%g)\n",
                        initial_time,plot_interval);
            }
        }
    }

    /* ------------------ Print some parameters --------------------- */
    if (master()) {
        fprintf(stdout,"Dam break problem\n");
        fprintf(stdout,"Using a %f x %f x %f grid of %d x %d x %d points\n",Lx,Ly,Lz,Nx,Ny,Nz);
        fprintf(stdout,"g = %f, rot_f = %f, rho_0 = %f, delta_rho = %f\n",g,rot_f,rho_0,delta_rho);
        fprintf(stdout,"Pycnocline half-width: h = %g\n",h_halfwidth);
        fprintf(stdout,"Pycnocline vertical shift %%: zeta = %g\n",pyc_asym);
        fprintf(stdout,"Pycnocline separation: zeta_p = %g\n",pyc_sep);
        fprintf(stdout,"Width of mixed region: L_mix = %g\n",Lmix);
        fprintf(stdout,"Height of mixed region: H_mix = %g\n",Hmix);
        fprintf(stdout,"Time between plots: %g s\n",plot_interval);
        fprintf(stdout,"Initial velocity perturbation: %g\n",perturb);
        fprintf(stdout,"Stably-stratified phase speed %g\n",c0);
        fprintf(stdout,"Buoyancy frequency squared %g\n",N2_max);
        fprintf(stdout,"Reynolds number %g\n",Re);
    }

    /* ------------------ Do stuff --------------------- */
    dambreak mycode; // Create an instantiated object of the above class
    // Create a flow-evolver that takes its settings from the above class
    FluidEvolve<dambreak> do_stuff(&mycode);
    // Initialize
    do_stuff.initialize();
    step_start_time = MPI_Wtime(); // beginning of simulation (after reading in data)
    t_startup = step_start_time - real_start_time;
    fprintf(stdout,"Start-up time: %.6g s.\n",t_startup);
    // Run until the end of time
    do_stuff.do_run(final_time);
    MPI_Finalize(); // Cleanly exit MPI
    return 0; // End the program
}
