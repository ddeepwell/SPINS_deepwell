/* Generic script for two layer fluid with zero initial velocity
   and no topography */

/* ------------------ Top matter --------------------- */

// Required headers
#include "../BaseCase.hpp"      // contains default class
#include "../Options.hpp"       // config-file parser
#include <random/normal.h>      // Blitz random number generator

using namespace ranlib;

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
S_EXP expan[3];

// Physical parameters
double g, rot_f, rho_0;     // gravity accel (m/s^2), Coriolis frequency (s^-1), reference density (kg/L)
double visco;               // viscosity (m^2/s)
double mu;                  // dynamic viscosity (kg/(m·s))
double kappa_rho;           // diffusivity of density (m^2/s)
double kappa_tracer;        // diffusivity of tracer (m^2/s)
// helpful constants
const int NUM_TRACER = 2;
const int RHO = 0;          // index for rho
const int TRCR = 1;         // index for tracer

// Stratification parameters
double delta_rho;           // change in density (kg/L)
double rho_z_center;        // vertical location of pycnocline (AS PERCENT OF TOTAL)
double rho_z_width;         // vertical width of rho (AS PERCENT OF TOTAL)

// Forcing parameters
double jet_strength;        // strength of jets (kg m /s^2)
double jet_t_period;        // period of jets in time (s)
double jet_x_period;        // period of jets in x 
double jet_z_period;        // period of jets in z 
double jet_x_center;        // center of jets in the x (m)
double jet_x_width;         // width of jets in the x (m)

// Tracer parameters
double tracer_on;			// edge of tracer column (m)
double tracer_off;			// edge of tracer column (m)
double tracer_transition;   // horizontal transition length (m)

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
bool compute_stress;        // Compute surface stresses?
bool compute_enstrophy;     // Compute Enstrophy?
bool compute_dissipation;   // Compute dissipation?
int itercount = 0;          // Iteration counter

// Writing options
int verbosity;
bool write_p; 
bool write_u;
bool write_v;
bool write_w;
bool write_rho;
bool write_tracer;
bool write_du;
bool write_dv;
bool write_dw;
bool write_drho;
bool write_dtracer;
bool write_ddu;
bool write_ddv;
bool write_ddw;
bool write_ddrho;
bool write_ddtracer;
bool write_vorticity;
bool write_enstrophy;
bool write_stress;
bool write_dissipation;


/* ------------------ Derived parameters --------------------- */

// Flow speed
double c0;

// Squared maximum buoyancy frequency if the initial stratification was stable
double N2_max;

// Reynolds number
double Re;

/* ------------------ Adjust the class --------------------- */

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

        /* Variables for Diagnostics */
        double max_u, max_v, max_w, max_rho, max_dye;
        double max_ke, ke_tot, pe_tot, diss_tot;

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
                    return kappa_tracer;
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
            if (master()) fprintf(stdout,"Initializing velocities\n");
            // if restarting
            if (restarting and !restart_from_dump) {
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
                            if ( Ny > 1 )
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

        /* Initialize the tracers (density and dyes) */
        void init_tracers(vector<DTArray *> & tracers) {
            if (master()) fprintf(stdout,"Initializing Density\n");
            DTArray & rho = *tracers[RHO];
            DTArray & tracer = *tracers[TRCR];
            assert(numtracers() == int(tracers.size()));

            if (restarting and !restart_from_dump) {
                init_tracer_restart("rho",rho);
                init_tracer_restart("tracer",tracer);
            } else if (restarting and restart_from_dump) {
                init_tracer_dump("rho",rho);
                init_tracer_dump("tracer",tracer);
            } else {
                // background stratification
                rho = -0.5*delta_rho*tanh((zz(kk)-rho_z_center*Lz)/(rho_z_width));
                // tracer profile
                tracer = 0.5*(1.0+tanh((xx(ii)-tracer_on )/tracer_transition));
                tracer -= 0.5*(1.0+tanh((xx(ii)-tracer_off)/tracer_transition));
                // Write the arrays
                write_array(rho,"rho",plotnum);
                write_array(tracer,"tracer",plotnum); 
            }
        }

        /* Forcing in the momentum equations */
        void forcing(double t, DTArray & u, DTArray & u_f,
                DTArray & v, DTArray & v_f, DTArray & w, DTArray & w_f,
                vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
            u_f = +rot_f*v;
            v_f = -rot_f*u;
            w_f = -g*(*tracers[RHO])  // tracer[RHO] = rho/rho_0 
                + jet_strength*sin(2*M_PI*jet_t_period*t)*exp(-pow((xx(ii)-jet_x_center)/jet_x_width,2))
                *sin(2*M_PI*jet_z_period*zz(kk)/Lz)*sin(2*M_PI*jet_x_period*xx(ii)/Lx);
            *tracers_f[RHO] = 0;
            *tracers_f[TRCR] =0;
        }

        void initialize_diagnostics_file() {
            if (master() and !restarting) {
                // create file for other diagnostics and write the column headers
                FILE * diagnos_file = fopen("diagnostics.txt","a");
                assert(diagnos_file);
                fprintf(diagnos_file,"Iter, Clock_time, Sim_time, "
                        "Max_U, Max_V, Max_W, "
                        "Max_KE, Total_KE, Total_PE, Total_dissipation, "
                        "Max_density, Max_tracer\n");
                fclose(diagnos_file);
            }
        }

        void write_diagnostics(double time) {
            if (master()) {
                /* add to the diagnostics file at each time step */
                FILE * diagnos_file = fopen("diagnostics.txt","a");
                assert(diagnos_file);
                fprintf(diagnos_file,"%d, %.12g, %.12f, "
                        "%.12g, %.12g, %.12g, "
                        "%.12g, %.12g, %.12g, %.12g, "
                        "%.12g, %.12g\n",
                        itercount,t_step,time,
                        max_u,max_v,max_w,
                        max_ke,ke_tot,pe_tot,diss_tot,
                        max_rho,max_dye);
                fclose(diagnos_file);
                /* and to the log file */
                fprintf(stdout,"[%d] (%.4g) %.4f: "
                        "%.4g %.4g %.4g "
                        "%.4g %.4g %.4g %.4g "
                        "%.4g %.4g\n",
                        itercount,t_step,time,
                        max_u,max_v,max_w,
                        max_ke,ke_tot,pe_tot,diss_tot,
                        max_rho,max_dye);
            }
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
                // initialize the diagnostic files
                initialize_diagnostics_file();
            }
            // update clocks
            if (master()) {
                clock_time = MPI_Wtime();
                t_step = clock_time - step_start_time;
            }

            /* Calculate and write out useful information */

            // total dissipation
            diss_tot = 0;
            if (compute_dissipation) {
                dissipation(u, v, w, *temp1, gradient_op, grid_type, Nx, Ny, Nz, mu);
                diss_tot = pssum(sum((*temp1)*
                            (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            }
            // Energy (PE assumes density is density anomaly)
            ke_tot = pssum(sum(0.5*rho_0*(u*u + v*v + w*w)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            pe_tot = pssum(sum(rho_0*(1+*tracers[RHO])*g*(zz(kk) - MinZ)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            // max of fields
            max_u = psmax(max(abs(u)));
            max_v = psmax(max(abs(v)));
            max_w = psmax(max(abs(w)));
            max_ke = psmax(max(0.5*rho_0*(u*u + v*v + w*w)*
                        (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
            max_rho = psmax(max(abs(*tracers[RHO])));
            max_dye = psmax(max(abs(*tracers[TRCR])));

            // write to the diagnostic file
            write_diagnostics(time);

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

            /* Write to disk if at correct time */
            if ((time - next_plot) > -1e-6*plot_interval) {
                plotnum++;
                t_step = MPI_Wtime(); // time just before write (for dump)
                // Write the arrays
                if (write_p)
                    write_array(pressure,"p",plotnum);
                if (write_u)
                    write_array(u,"u",plotnum);
                if ( (Ny > 1 or rot_f != 0) and write_v == true)
                    write_array(v,"v",plotnum);
                if (write_w)
                    write_array(w,"w",plotnum);
                if (write_rho)
                    write_array(*tracers[RHO],"rho",plotnum);
                if (write_tracer)
                    write_array(*tracers[TRCR],"tracer",plotnum);
                if (write_du) {
                    // u derivatives
                    find_expansion(grid_type, expan, "u");
                    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"u_x",plotnum);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"u_y",plotnum);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"u_z",plotnum);
                }
                if ( (Ny > 1 or rot_f != 0) and write_dv == true) {
                    // v derivatives
                    find_expansion(grid_type, expan, "v");
                    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"v_x",plotnum);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"v_y",plotnum);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"v_z",plotnum);
                }
                if (write_dw) {
                    // w derivatives
                    find_expansion(grid_type, expan, "w");
                    gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"w_x",plotnum);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"w_y",plotnum);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"w_z",plotnum);
                }
                if (write_drho) {
                    // Density derivatives
                    find_expansion(grid_type, expan, "rho");
                    gradient_op->setup_array(tracers[RHO],expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"rho_x",plotnum);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"rho_y",plotnum);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"rho_z",plotnum);
                }
                if (write_dtracer) {
                    // tracer derivatives
                    find_expansion(grid_type, expan, "tracer");
                    gradient_op->setup_array(tracers[TRCR],expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"tracer_x",plotnum);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"tracer_y",plotnum);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"tracer_z",plotnum);
                }
                if (write_ddu) {
                    // second u derivatives
                    DTArray *ux = alloc_array(Nx,Ny,Nz);
                    DTArray *uy = alloc_array(Nx,Ny,Nz);
                    DTArray *uz = alloc_array(Nx,Ny,Nz);
                    find_expansion(grid_type, expan, "u");
                    gradient_op->setup_array(&u,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(ux,false);
                    gradient_op->get_dy(uy,false);
                    gradient_op->get_dz(uz,false);
                    find_expansion(grid_type, expan, "u_x");
                    gradient_op->setup_array(ux,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"u_xx",plotnum);
                    find_expansion(grid_type, expan, "u_y");
                    gradient_op->setup_array(uy,expan[0],expan[1],expan[2]);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"u_yy",plotnum);
                    find_expansion(grid_type, expan, "u_z");
                    gradient_op->setup_array(uz,expan[0],expan[1],expan[2]);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"u_zz",plotnum);
                    delete ux, uy, uz;
                }
                if (write_ddv) {
                    // second v derivatives
                    DTArray *vx = alloc_array(Nx,Ny,Nz);
                    DTArray *vy = alloc_array(Nx,Ny,Nz);
                    DTArray *vz = alloc_array(Nx,Ny,Nz);
                    find_expansion(grid_type, expan, "v");
                    gradient_op->setup_array(&v,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(vx,false);
                    gradient_op->get_dy(vy,false);
                    gradient_op->get_dz(vz,false);
                    find_expansion(grid_type, expan, "v_x");
                    gradient_op->setup_array(vx,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"v_xx",plotnum);
                    find_expansion(grid_type, expan, "v_y");
                    gradient_op->setup_array(vy,expan[0],expan[1],expan[2]);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"v_yy",plotnum);
                    find_expansion(grid_type, expan, "v_z");
                    gradient_op->setup_array(vz,expan[0],expan[1],expan[2]);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"v_zz",plotnum);
                    delete vx, vy, vz;
                }
                if (write_ddw) {
                    // second w derivatives
                    DTArray *wx = alloc_array(Nx,Ny,Nz);
                    DTArray *wy = alloc_array(Nx,Ny,Nz);
                    DTArray *wz = alloc_array(Nx,Ny,Nz);
                    find_expansion(grid_type, expan, "w");
                    gradient_op->setup_array(&w,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(wx,false);
                    gradient_op->get_dy(wy,false);
                    gradient_op->get_dz(wz,false);
                    find_expansion(grid_type, expan, "w_x");
                    gradient_op->setup_array(wx,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"w_xx",plotnum);
                    find_expansion(grid_type, expan, "w_y");
                    gradient_op->setup_array(wy,expan[0],expan[1],expan[2]);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"w_yy",plotnum);
                    find_expansion(grid_type, expan, "w_z");
                    gradient_op->setup_array(wz,expan[0],expan[1],expan[2]);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"w_zz",plotnum);
                    delete wx, wy, wz;
                }
                if (write_ddrho) {
                    // second density derivatives
                    DTArray *rhox = alloc_array(Nx,Ny,Nz);
                    DTArray *rhoy = alloc_array(Nx,Ny,Nz);
                    DTArray *rhoz = alloc_array(Nx,Ny,Nz);
                    find_expansion(grid_type, expan, "rho");
                    gradient_op->setup_array(tracers[RHO],expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(rhox,false);
                    gradient_op->get_dy(rhoy,false);
                    gradient_op->get_dz(rhoz,false);
                    find_expansion(grid_type, expan, "rho_x");
                    gradient_op->setup_array(rhox,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"rho_xx",plotnum);
                    find_expansion(grid_type, expan, "rho_y");
                    gradient_op->setup_array(rhoy,expan[0],expan[1],expan[2]);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"rho_yy",plotnum);
                    find_expansion(grid_type, expan, "rho_z");
                    gradient_op->setup_array(rhoz,expan[0],expan[1],expan[2]);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"rho_zz",plotnum);
                    delete rhox, rhoy, rhoz;
                }
                if (write_ddtracer) {
                    // second tracer derivatives
                    DTArray *tracerx = alloc_array(Nx,Ny,Nz);
                    DTArray *tracery = alloc_array(Nx,Ny,Nz);
                    DTArray *tracerz = alloc_array(Nx,Ny,Nz);
                    find_expansion(grid_type, expan, "tracer");
                    gradient_op->setup_array(tracers[TRCR],expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(tracerx,false);
                    gradient_op->get_dy(tracery,false);
                    gradient_op->get_dz(tracerz,false);
                    find_expansion(grid_type, expan, "tracer_x");
                    gradient_op->setup_array(tracerx,expan[0],expan[1],expan[2]);
                    gradient_op->get_dx(temp1,false);
                    write_array(*temp1,"tracer_xx",plotnum);
                    find_expansion(grid_type, expan, "tracer_y");
                    gradient_op->setup_array(tracery,expan[0],expan[1],expan[2]);
                    gradient_op->get_dy(temp1,false);
                    write_array(*temp1,"tracer_yy",plotnum);
                    find_expansion(grid_type, expan, "tracer_z");
                    gradient_op->setup_array(tracerz,expan[0],expan[1],expan[2]);
                    gradient_op->get_dz(temp1,false);
                    write_array(*temp1,"tracer_zz",plotnum);
                    delete tracerx, tracery, tracerz;
                }
                if (write_vorticity) {
                    // write vorticity
                    compute_vort_x(v, w, *temp1, gradient_op, grid_type);
                    write_array(*temp1,"vortx",plotnum);
                    compute_vort_y(u, w, *temp1, gradient_op, grid_type);
                    write_array(*temp1,"vorty",plotnum);
                    compute_vort_z(u, v, *temp1, gradient_op, grid_type);
                    write_array(*temp1,"vortv",plotnum);
                }
                if (write_enstrophy) {
                    enstrophy_density(u, v, w, *temp1, gradient_op, grid_type, Nx, Ny, Nz);
                    write_array(*temp1,"enst",plotnum);
                }
                if (write_stress) {
                    // not complete
                }
                if (write_dissipation) {
                    dissipation(u, v, w, *temp1, gradient_op, grid_type, Nx, Ny, Nz, visco);
                    write_array(*temp1,"diss",plotnum);
                }

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
        }

        // User specified variables to dump
        void write_variables(DTArray & u,DTArray & v, DTArray & w,
                vector<DTArray *> & tracers) {
            write_array(u,"u.dump",-1);
            write_array(v,"v.dump",-1);
            write_array(w,"w.dump",-1);
            write_array(*tracers[RHO],"rho.dump",-1);
            write_array(*tracers[TRCR],"tracer.dump",-1);
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
    add_option("rho_0",&rho_0,1000.0,"Reference density");
    add_option("visco",&visco,0.0,"Viscosity");
    add_option("kappa_rho",&kappa_rho,0.0,"Diffusivity of density");	
    add_option("kappa_tracer",&kappa_tracer,0.0,"Diffusivity of tracer");

    option_category("Stratification parameters");
    add_option("delta_rho",&delta_rho,"density difference");
    add_option("rho_z_center",&rho_z_center,"density difference");
    add_option("rho_z_width",&rho_z_width,"vertical transition half-width");

    option_category("Forcing parameters");
    add_option("jet_strength",&jet_strength,"Strength of jets");
    add_option("jet_t_period",&jet_t_period,"period of temporal oscillations");
    add_option("jet_x_period",&jet_x_period,"period of x oscillations");
    add_option("jet_z_period",&jet_z_period,"period of z oscillations");
    add_option("jet_x_center",&jet_x_center,"x center of jets");
    add_option("jet_x_width",&jet_x_width,"width of jets in x");

    option_category("Tracer parameters");
    add_option("tracer_on",&tracer_on,"start of tracer column");
    add_option("tracer_off",&tracer_off,"end of tracer column");
    add_option("tracer_transition",&tracer_transition,"Horizontal transition half-width");

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

    option_category("Writting options");
    add_option("verbosity",&verbosity,1,"Level of outputs desired (0-4)");
    add_option("write_p",&write_p,false,"Enable the outputting of the pressure");
    add_option("write_u",&write_u,false,"Enable the outputting of u");
    add_option("write_v",&write_v,false,"Enable the outputting of v");
    add_option("write_w",&write_w,false,"Enable the outputting of w");
    add_option("write_rho",&write_rho,false,"Enable the outputting of rho");
    add_option("write_tracer",&write_tracer,false,"Enable the outputting of tracer");
    add_option("write_du",&write_du,false,"Enable the outputting of the derivatives of u");
    add_option("write_dv",&write_dv,false,"Enable the outputting of the derivatives of v");
    add_option("write_dw",&write_dw,false,"Enable the outputting of the derivatives of w");
    add_option("write_drho",&write_drho,false,"Enable the outputting of the derivatives of rho");
    add_option("write_dtracer",&write_dtracer,false,"Enable the outputting of the derivatives of tracer");
    add_option("write_ddu",&write_ddu,false,"Enable the outputting of the second derivatives of u");
    add_option("write_ddv",&write_ddv,false,"Enable the outputting of the second derivatives of v");
    add_option("write_ddw",&write_ddw,false,"Enable the outputting of the second derivatives of w");
    add_option("write_ddrho",&write_ddrho,false,"Enable the outputting of the second derivatives of rho");
    add_option("write_ddtracer",&write_ddtracer,false,"Enable the outputting of the second derivatives of tracer");
    add_option("write_vorticity",&write_vorticity,false,"Enable the outputting of the vorticity");
    add_option("write_enstrophy",&write_enstrophy,false,"Enable the outputting of the enstrophy");
    add_option("write_stress",&write_stress,false,"Enable the outputting of the stress");
    add_option("write_dissipation",&write_dissipation,false,"Enable the outputting of the viscous dissipation");

    option_category("Other options");
    add_option("perturb",&perturb,0.0,"Initial perturbation in velocity");
    add_option("compute_stress",&compute_stress,false,"Calculate the top and bottom stresses?");
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

    // Check verbosity level
    if (verbosity == 1) {
        // base level
        write_p =           false;
        write_u =           true;
        write_v =           true;
        write_w =           true;
        write_rho =         true;
        write_tracer =      true;
        write_du =          false;
        write_dv =          false;
        write_dw =          false;
        write_drho =        false;
        write_dtracer =     false;
        write_ddu =         false;
        write_ddv =         false;
        write_ddw =         false;
        write_ddrho =       false;
        write_ddtracer =    false;
        write_vorticity =   false;
        write_stress =      false;
        write_enstrophy =   false;
        write_dissipation = false;
    } else if (verbosity == 2) {
        // 1 + vorticity
        write_p =           false;
        write_u =           true;
        write_v =           true;
        write_w =           true;
        write_rho =         true;
        write_tracer =      true;
        write_du =          false;
        write_dv =          false;
        write_dw =          false;
        write_drho =        false;
        write_dtracer =     false;
        write_ddu =         false;
        write_ddv =         false;
        write_ddw =         false;
        write_ddrho =       false;
        write_ddtracer =    false;
        write_vorticity =   true;
        write_stress =      false;
        write_enstrophy =   false;
        write_dissipation = false;

    } else if (verbosity == 3) {
        // 1 + derived
        write_p =           false;
        write_u =           true;
        write_v =           true;
        write_w =           true;
        write_rho =         true;
        write_tracer =      true;
        write_du =          false;
        write_dv =          false;
        write_dw =          false;
        write_drho =        false;
        write_dtracer =     false;
        write_ddu =         false;
        write_ddv =         false;
        write_ddw =         false;
        write_ddrho =       false;
        write_ddtracer =    false;
        write_vorticity =   true;
        write_stress =      true;
        write_enstrophy =   true;
        write_dissipation = true;

    } else if (verbosity == 4) {
        // 1 + derivatives
        write_p =           false;
        write_u =           true;
        write_v =           true;
        write_w =           true;
        write_rho =         true;
        write_tracer =      true;
        write_du =          true;
        write_dv =          true;
        write_dw =          true;
        write_drho =        true;
        write_dtracer =     true;
        write_ddu =         true;
        write_ddv =         true;
        write_ddw =         true;
        write_ddrho =       true;
        write_ddtracer =    true;
        write_vorticity =   false;
        write_stress =      false;
        write_enstrophy =   false;
        write_dissipation = false;
    } else if (verbosity == 5) {
        // 'ery thing
        write_p =           true;
        write_u =           true;
        write_v =           true;
        write_w =           true;
        write_rho =         true;
        write_tracer =      true;
        write_du =          true;
        write_dv =          true;
        write_dw =          true;
        write_drho =        true;
        write_dtracer =     true;
        write_ddu =         true;
        write_ddv =         true;
        write_ddw =         true;
        write_ddrho =       true;
        write_ddtracer =    true;
        write_vorticity =   true;
        write_stress =      true;
        write_enstrophy =   true;
        write_dissipation = true;
    } 
    else {
        // user specified
    }

    // adjust time if starting from a dump
    if (restart_from_dump) {
        adjust_for_dump(restarting, restart_time, restart_sequence,
                final_time, compute_time, avg_write_time, Nx, Ny, Nz);
    }

    // check the restart sequence
    check_restart_sequence(restarting, restart_sequence, initial_time, restart_time, plot_interval);

    // parse expansion types
    get_boundary_conditions(xgrid_type, ygrid_type, zgrid_type, intype_x, intype_y, intype_z);
    // vector of string types
    grid_type[0] = xgrid_type;
    grid_type[1] = ygrid_type;
    grid_type[2] = zgrid_type;

    // check if this is mapped - it shouldn't be
    if (mapped) {
        if (master())
            fprintf(stderr,"This case file is for unmapped cases only. "
                    "Change spins.conf, or use a different case file.\n");
        MPI_Finalize(); exit(1);
    }

    // adjust Ly for 2D
    if (Ny==1 and Ly!=1.0) {
        Ly = 1.0;
        if (master())
            fprintf(stdout,"Simulation is 2 dimensional, "
                    "Ly has been changed to 1.0 for normalization.\n");
    }

    /* ------------------ Derived parameters --------------------- */

    // Dynamic viscosity
    mu = visco*rho_0;
    // Mode-2 wave speed
    c0 = 0;//0.5*sqrt(g*rho_z_width*delta_rho/rho_0);
    // Maximum buoyancy frequency (squared) if the initial stratification was stable
    N2_max = g*delta_rho/(2*rho_z_width);       // delta_rho is already scaled by rho_0
    // Reynolds number
    //Re = c0*rho_z_width/visco;
    // Maximum time step
    dt_max = 0.5/sqrt(N2_max);

    /* ------------------ Print some parameters --------------------- */
    if (master()) {
        fprintf(stdout,"Cenote jet problem\n");
        fprintf(stdout,"Using a %f x %f x %f grid of %d x %d x %d points\n",Lx,Ly,Lz,Nx,Ny,Nz);
        fprintf(stdout,"g = %f, rot_f = %f, rho_0 = %f\n",g,rot_f,rho_0);
        fprintf(stdout,"Time between plots: %g s\n",plot_interval);
        fprintf(stdout,"Initial velocity perturbation: %g\n",perturb);
        fprintf(stdout,"Filter cutoff = %f, order = %f, strength = %f\n",f_cutoff,f_order,f_strength);
        fprintf(stdout,"Density difference: delta_rho = %g\n",delta_rho);
        fprintf(stdout,"Pycnocline half-width: h = %g\n",rho_z_width);
        fprintf(stdout,"Stably-stratified phase speed %g\n",c0);
        fprintf(stdout,"Buoyancy frequency squared %g\n",N2_max);
        //fprintf(stdout,"Reynolds number %g\n",Re);
    }

    /* ------------------ Do stuff --------------------- */
    dambreak mycode; // Create an instance of the above class
    FluidEvolve<dambreak> do_stuff(&mycode);
    do_stuff.initialize();
    step_start_time = MPI_Wtime(); // beginning of simulation (after reading in data)
    startup_time = step_start_time - real_start_time;
    if (master()) fprintf(stdout,"Start-up time: %.6g s.\n",startup_time);
    // Run until the end of time
    do_stuff.do_run(final_time);
    MPI_Finalize();
    return 0;
}
