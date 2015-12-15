/* vortex_reader.cpp -- 
 *
 * general case for looking at the evolution of large-scale
 * (f-plane) 3D systems. It was specifically designed to study
 * 3D vortices.
 *
 * Input data and configuration provided at runtime
 * via a configuration file (spins.conf). See config_vortex.py 
 *
 */

// Include the necessary packakges
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
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <boost/program_options.hpp> 
namespace po = boost::program_options;

using std::string;


/* Domain Parameters  */
int      Nx,   Ny,   Nz;   // Number of points in x, y, z
double   Lx,   Ly,   Lz,   // Grid lengths of x, y, z 
         MinX, MinY, MinZ; // Minimum x/y/z points
bool mapped;               // Is the domain mapped?


/* Pertinent (tpyically input) filenames */
string xgrid_filename,
       ygrid_filename,
       zgrid_filename,
       u_filename,
       v_filename,
       w_filename,
       rho_filename,
       tracer_filename,
       bg_b_filename;

/* Physical parameters */
double g, rot_f, N0, vel_mu, dens_kappa, tracer_kappa, tracer_g,
       alpha, beta, H; 

/* Numerical Parameters */
double max_dt=100.0; // Initialize as something, is computed later
double perturb = 0;  // Amount of perturbation to add
bool tracer;         // Is there a passive tracer?

/* Writeout parameters */
double final_time, plot_interval;
int plot_write_ratio;
double initial_time;
bool compute_norms;
int slice_write = 0;

// Histogram parameters
int num_bins = 100;
double hist_bins[100]; 
double hist_vols[100]; 
double hist_cdf[100]; 
double hist_new_bins[100]; 

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

// Dump parameters
double t_step;
double real_start_time;
double compute_time;
bool restart_from_dump = false;
double total_run_time;
double avg_write_time;

////
//// Declarations for chains and slices
////

int X_ind = 0;
int Y_ind = 1;
int Z_ind = 2;

// Initialize slice files 
int num_slices[3];
double * slice_coords[3];
int slice_write_count, prev_slice_write_count;
MPI_File * u_slice_files[3];        // Create a pair of files for each variable
MPI_File * u_slice_final_files[3];  //   that you want to write to a slice.
MPI_File * v_slice_files[3];        //   Each variable will be written to every slice.
MPI_File * v_slice_final_files[3];
MPI_File * w_slice_files[3];
MPI_File * w_slice_final_files[3];
MPI_File * b_slice_files[3];
MPI_File * b_slice_final_files[3];
MPI_File * e_slice_files[3];
MPI_File * e_slice_final_files[3];

// Initialize chain files
int num_chains[3];
double ** chain_coords[3];
int chain_write_count, prev_chain_write_count;
double * x_chain_data_buffer;       // To avoid repeatedly allocating these,
double * y_chain_data_buffer;       //   declare them now.
double * z_chain_data_buffer;
MPI_File * w_chain_files[3];        // Create a pair of files for each variable
MPI_File * w_chain_final_files[3];  //   that you want to write to a chain.
MPI_File * b_chain_files[3];        //   Each variable will be written to every chain.
MPI_File * b_chain_final_files[3];

/* ------------------------------- */
/* Begin definition of userControl */
/* ------------------------------- */
class userControl : public BaseCase {
    public:

        /* Variables for tracking progress  */
        int plotnum, itercount, lastplot, last_writeout;
        bool plot_now;
        double nextplot;

        /* Variables for timing the simulation */
        double start_rho_spread, step_dur;
        double clock_time, start_time;

        /* Functions that return domain information */
        int     size_x()       const { return Nx; }
        int     size_y()       const { return Ny; }
        int     size_z()       const { return Nz; }
        double  length_x()     const { return Lx; }
        double  length_y()     const { return Ly; }
        double  length_z()     const { return Lz; }
        DIMTYPE type_x()       const { return intype_x; }
        DIMTYPE type_y()       const { return intype_y; }
        DIMTYPE type_default() const { return intype_z; }
        bool    is_mapped()    const { return mapped;   }

        // Function to return restart information
        int  get_restart_sequence()  const { return restart_sequence; }

        // Grids
        Array<double,1> xgrid, ygrid, zgrid;

        /* Variables for Diagnostics */ 
        DTArray* e_pv;   // Ertel PV
        DTArray* bg_b;   // Background buoyancy
        double KE, PE, rho0, b_pert_norm, // Some diagnostics
               del_b, norm_u, norm_v, norm_w, BPE, APE,
               hist_var, curr_val, old_var, tmp_double;
        double * chain_data_buffer;
        double * slice_data_buffer;
        int II, JJ, KK, Ix, Iy, Iz;
        double min_zz, avg_rho;
        double Vol() const  { return (Lx/Nx)*(Ly/Ny)*(Lz/Nz);}
        double Vol_PE() const  { return (Lx/Nx)*(Ly/Ny);}
        FILE * diagnostics;  // File for writing diagnostics
        FILE * plottimes;    // File for tracking output times
        FILE * step_times;   // File for tracking how long each step takes

        // Files for chain outputs
        MPI_File ** x_chain_final;
        MPI_File ** y_chain_final;
        MPI_File ** z_chain_final;

        // Files for slice outputs
        MPI_File ** xy_slice_final;
        MPI_File ** xz_slice_final;
        MPI_File ** yz_slice_final;

        int my_rank, num_procs, pointflag;

        /* Function to access viscosity */
        double get_visco() const {
            return vel_mu;
        }

        /* Function to access diffusivity */
        double get_diffusivity(int t) const {
            if (t == 0) return dens_kappa; 
         if (t == 1) return tracer_kappa;
         else assert(0 && "Invalid tracer number!");
        }

        /* Function to return the start time */
        double init_time() const { 
            return initial_time;
        }

        /* Function to create the grids */
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

            // Write out the grids 
            write_array(xg,"xgrid");
            write_array(yg,"ygrid");
            write_array(zg,"zgrid");
        }

        /* We have an active tracer, namely density */
        int numActive() const { return 1; }

        /* We're given a passive tracer to advect */
        int numPassive() const {
            if (tracer) return 1;
            else return 0;
        }

        /* Timestep-check function.  This (along with "write everything" outputs) should
           really be bumped into the BaseCase */
        double check_timestep(double intime, double now) {
            if (intime < 1e-9) {
                /* Timestep's too small, somehow stuff is blowing up */
                if (master()) fprintf(stderr,"Tiny timestep (%e), aborting\n",intime);
                return -1;
                //FJP: intime sets time step
            } else if (intime > max_dt) {
                /* Cap the maximum timestep size */
                intime = max_dt;
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

        void initialize_diagnostics_files() {
            // Create empty files for storage and add header when necessary
            if (master()) {
                fprintf(stdout,"Initializing diagnostics files. \n");

                // File to track plot times
                if (!restarting) plottimes = fopen("plot_times.txt","w");
                else             plottimes = fopen("plot_times.txt","a");
                assert(plottimes);
                fclose(plottimes);

                // File to track diagnostics
                if (!restarting) {
                    diagnostics = fopen("diagnostics.txt","w");
                    assert(diagnostics);
                    fprintf(diagnostics,"time, iteration, norm_u, norm_v, norm_w, del_b, KE, APE, BPE, step_time");
                    if (compute_norms)  fprintf(diagnostics,", b_pert_norm");
                    fprintf(diagnostics,"\n");
                }
                else {
                    diagnostics = fopen("diagnostics.txt","a");
                    assert(diagnostics);
                }
                fclose(diagnostics);

                // File to track step times
                if (!restarting) step_times = fopen("step_times.txt","w");
                else             step_times = fopen("step_times.txt","a");
                assert(step_times);
                fprintf(step_times, "iter, step_time\n");
                fclose(step_times);
            }
        }

        void write_diagnostics(double sim_time) {
            if (master()) {

                // Update diagnostics file
                diagnostics = fopen("diagnostics.txt","a");
                assert(diagnostics);
                fprintf(diagnostics,"%f, %d, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g, %.12g",
                        sim_time,itercount,norm_u,norm_v,norm_w,del_b,KE,APE,BPE,step_dur);
                if (compute_norms) fprintf(diagnostics, ", %.12g", b_pert_norm);
                fprintf(diagnostics, "\n");
                fclose(diagnostics);

                // Write step information to stdout
                fprintf(stdout,"t = %f [ii = %d]: norm(u,v,w) = (%.2g, %.2g, %.2g) : delta(b) = %.2g : (KE,APE,BPE,PE) = (%.3g,%.3g,%.2g,%.2g)",
                        sim_time,itercount,norm_u,norm_v,norm_w,del_b,KE,APE,BPE,PE);
                if (compute_norms) fprintf(stdout, " : epv_pert = %.2g",b_pert_norm);
                fprintf(stdout, "\n");
            }
        }

        // Initialize the chain files
        void initialize_chains() {
            // This function initializes all of the files required for writing chains.
            // To add more variable outputs, simple replicate these samples.

            // Initialize the w chain files
            initialize_chain_tmps("w", w_chain_files, chain_coords, num_chains);
            initialize_chain_finals("w", w_chain_final_files, chain_coords, num_chains);

            // Initialize the b chain files
            initialize_chain_tmps("b", b_chain_files, chain_coords, num_chains);
            initialize_chain_finals("b", b_chain_final_files, chain_coords, num_chains);
        }

        // User-specified "chain" (1D) outputs
        void write_chains(DTArray & w, vector<DTArray *> & tracer) {
            // This function writes to the chain files. At the moment, this happens
            // at every output.

            write_chains_2(w, w_chain_files, chain_coords, num_chains, Nx, Ny, Nz, chain_write_count,
                    x_chain_data_buffer, y_chain_data_buffer, z_chain_data_buffer);
            write_chains_2(*tracer[0], b_chain_files, chain_coords, num_chains, Nx, Ny, Nz, chain_write_count,
                    x_chain_data_buffer, y_chain_data_buffer, z_chain_data_buffer);

        }

        // Re-open chain files after stitching
        void reopen_chains() {
            // After the chains have been 'stitched' (for restart-proofing)
            // re-open the temporary files.

            initialize_chain_tmps("w", w_chain_files, chain_coords, num_chains); // Reopen w chain files
            initialize_chain_tmps("b", b_chain_files, chain_coords, num_chains); // Reopen b chain files
        }

        void close_chains() {
            // When finished, close the chains.
            // Temp files will automatically be deleted.
            for (int II = 0; II < 3; II++) {
                for (int JJ = 0; JJ < num_chains[II]; JJ++) {
                    MPI_File_close(&w_chain_files[II][JJ]);
                    MPI_File_close(&w_chain_final_files[II][JJ]);

                    MPI_File_close(&b_chain_files[II][JJ]);
                    MPI_File_close(&b_chain_final_files[II][JJ]);
                }
            }
        }

        // Stitch together the chain data into the final output files
        void stitch_chains(DTArray & w) {
            // To make the code more restart-proof, the chains are written to temp files
            // until a major (3D) output is reached, at which point the temp file is appended
            // to the full chain output.

            stitch_chains_2("w", w_chain_files, w_chain_final_files, num_chains,
                    Nx, Ny, Nz, chain_coords, chain_write_count, prev_chain_write_count,
                    w.lbound(firstDim), w.ubound(firstDim));

            stitch_chains_2("b", b_chain_files, b_chain_final_files, num_chains,
                    Nx, Ny, Nz, chain_coords, chain_write_count, prev_chain_write_count,
                    w.lbound(firstDim), w.ubound(firstDim));
        }

        void initialize_slices() {

            // Initialize u slice files
            initialize_slice_tmps("u", u_slice_files, slice_coords, num_slices);
            initialize_slice_finals("u", u_slice_final_files, slice_coords, num_slices);

            // Initialize v slice files
            initialize_slice_tmps("v", v_slice_files, slice_coords, num_slices);
            initialize_slice_finals("v", v_slice_final_files, slice_coords, num_slices);

            // Initialize w slice files
            initialize_slice_tmps("w", w_slice_files, slice_coords, num_slices);
            initialize_slice_finals("w", w_slice_final_files, slice_coords, num_slices);

            // Initialize b slice files
            initialize_slice_tmps("b", b_slice_files, slice_coords, num_slices);
            initialize_slice_finals("b", b_slice_final_files, slice_coords, num_slices);

            // Initialize epv slice files
            initialize_slice_tmps("epv", e_slice_files, slice_coords, num_slices);
            initialize_slice_finals("epv", e_slice_final_files, slice_coords, num_slices);
        }

        // User-specified "slice" (2D) outputs
        void write_slices(DTArray &u, DTArray &v, DTArray &w, vector<DTArray *> & tracer, DTArray *epv) {

            write_slices_2(u, u_slice_files, slice_coords, num_slices, Nx, Ny, Nz, slice_write_count);
            write_slices_2(v, v_slice_files, slice_coords, num_slices, Nx, Ny, Nz, slice_write_count);
            write_slices_2(w, w_slice_files, slice_coords, num_slices, Nx, Ny, Nz, slice_write_count);
            write_slices_2(*tracer[0], b_slice_files, slice_coords, num_slices, Nx, Ny, Nz, slice_write_count);
            write_slices_2(*epv, e_slice_files, slice_coords, num_slices, Nx, Ny, Nz, slice_write_count);

        }

        // Process the slices stitch together the slice and chain information when necessary
        // Afterwards, re-open the appropriate files, but wipe them clean.
        void stitch_slices(DTArray & w) {

            stitch_slices_2("u", u_slice_files, u_slice_final_files, num_slices,
                    Nx, Ny, Nz, slice_coords, slice_write_count, prev_slice_write_count,
                    w.lbound(firstDim), w.ubound(firstDim));

            stitch_slices_2("v", v_slice_files, v_slice_final_files, num_slices,
                    Nx, Ny, Nz, slice_coords, slice_write_count, prev_slice_write_count,
                    w.lbound(firstDim), w.ubound(firstDim));

            stitch_slices_2("w", w_slice_files, w_slice_final_files, num_slices,
                    Nx, Ny, Nz, slice_coords, slice_write_count, prev_slice_write_count,
                    w.lbound(firstDim), w.ubound(firstDim));

            stitch_slices_2("b", b_slice_files, b_slice_final_files, num_slices,
                    Nx, Ny, Nz, slice_coords, slice_write_count, prev_slice_write_count,
                    w.lbound(firstDim), w.ubound(firstDim));

            stitch_slices_2("epv", e_slice_files, e_slice_final_files, num_slices,
                    Nx, Ny, Nz, slice_coords, slice_write_count, prev_slice_write_count,
                    w.lbound(firstDim), w.ubound(firstDim));
        }

        void reopen_slices() {

            initialize_slice_tmps("u", u_slice_files, slice_coords, num_slices);
            initialize_slice_tmps("v", v_slice_files, slice_coords, num_slices);
            initialize_slice_tmps("w", w_slice_files, slice_coords, num_slices);
            initialize_slice_tmps("b", b_slice_files, slice_coords, num_slices);
            initialize_slice_tmps("epv", e_slice_files, slice_coords, num_slices);

        }

        void close_slices() {
            for (int II = 0; II < 3; II++) {
                for (int JJ = 0; JJ < num_slices[II]; JJ++) {
                    MPI_File_close(&u_slice_files[II][JJ]);
                    MPI_File_close(&u_slice_final_files[II][JJ]);

                    MPI_File_close(&v_slice_files[II][JJ]);
                    MPI_File_close(&v_slice_final_files[II][JJ]);

                    MPI_File_close(&w_slice_files[II][JJ]);
                    MPI_File_close(&w_slice_final_files[II][JJ]);

                    MPI_File_close(&b_slice_files[II][JJ]);
                    MPI_File_close(&b_slice_final_files[II][JJ]);

                    MPI_File_close(&e_slice_files[II][JJ]);
                    MPI_File_close(&e_slice_final_files[II][JJ]);
                }
            }
        }

        /* What analysis should be done after each timestep */
        void analysis(double sim_time, DTArray & u, DTArray & v, DTArray & w,
                vector<DTArray *> tracer, DTArray & pressure) {

            // Increase the itercount
            itercount = itercount + 1;
            rho0   = 1000;

            // Do all of this stuff on the first iteration only
            if (itercount == 1) {
                
                // Determine rank information
                MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
                MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

                // For the moment, hard-code the chain and slice
                // information until we can sort out how to include
                // then in the spins.conf.

                num_chains[X_ind] = 0; // Number of x-chains
                num_chains[Y_ind] = 1; // Number of y-chains
                num_chains[Z_ind] = 1; // Number of z-chains

                x_chain_data_buffer = new double[Nx/num_procs];
                y_chain_data_buffer = new double[Ny];
                z_chain_data_buffer = new double[Nz];

                chain_coords[X_ind] = new double* [3];
                chain_coords[Y_ind] = new double* [3];
                chain_coords[Z_ind] = new double* [3];
                chain_coords[X_ind][0] = new double[num_chains[X_ind]]; // y coord for x-chains
                chain_coords[X_ind][1] = new double[num_chains[X_ind]]; // z coord for x-chains
                chain_coords[Y_ind][0] = new double[num_chains[Y_ind]]; // x coord for y-chains
                chain_coords[Y_ind][1] = new double[num_chains[Y_ind]]; // z coord for y-chains
                chain_coords[Z_ind][0] = new double[num_chains[Z_ind]]; // x coord for z-chains
                chain_coords[Z_ind][1] = new double[num_chains[Z_ind]]; // y coord for z-chains

                chain_coords[Y_ind][0][0] = 0.5;
                chain_coords[Y_ind][1][0] = 0.5;

                chain_coords[Z_ind][0][0] = 0.5;
                chain_coords[Z_ind][1][0] = 0.5;

                num_slices[X_ind] = 1; // Number of yz-slices
                num_slices[Y_ind] = 0; // Number of xz-slices
                num_slices[Z_ind] = 2; // Number of xy-slices

                slice_coords[X_ind] = new double[num_slices[X_ind]]; // x coord for yz-slices
                slice_coords[Y_ind] = new double[num_slices[Y_ind]]; // y coord for yz-slices
                slice_coords[Z_ind] = new double[num_slices[Z_ind]]; // z coord for xy-slices

                slice_coords[X_ind][0] = 0.5; 
                slice_coords[Z_ind][0] = 0.4;
                slice_coords[Z_ind][1] = 0.5;

                // Create the files for writing slices and chains
                initialize_chains();
                initialize_slices();
                chain_write_count = 0;
                prev_chain_write_count = 0;
                slice_write_count = 0;
                prev_chain_write_count = 0;

                // If compute norms, then intialize the field
                if (compute_norms) {
                    bg_b = alloc_array(Nx,Ny,Nz);
                    read_array(*bg_b,bg_b_filename.c_str(),Nx,Ny,Nz);
                }

                // Begin tracking the time
                if (master()) start_time = MPI_Wtime();

                // Initial files for diagnostic outputs
                if (!restarting) initialize_diagnostics_files();

                // Since analysis doesn't run before the first
                // step, we'll have to output after the first step
                // Since the first (Euler) step is short, 
                // there should't be much of a change.
                if (!restarting) {
                    if (master()) fprintf(stdout,"Writing epv.0\n");
                    ertel_pv(u, v, w, *tracer[0], e_pv, rot_f, N0,
                            Lx, Ly, Lz, Nx, Ny, Nz,
                            type_x(), type_y(), type_z());

                    //write_array(*e_pv,"epv",0);
                    write_array(*e_pv,"epv",0);
                }

                // Determine last plot if restarting from the dump case
                if (restart_from_dump){
                    lastplot = restart_sequence*plot_interval;    
                }
            }

            // Compute how long the last step took
            if (master()) {
                clock_time = MPI_Wtime();
                step_dur = clock_time - start_time;
                step_times = fopen("step_times.txt","a");
                assert(step_times);
                fprintf(step_times, "%d,%.12g\n",itercount,step_dur);
                fclose(step_times);
            }

            // Approximate the APE with a hisotgram method
            // Store rho in e_pv
            //(*e_pv)(ii,jj,kk)  = rho0/g*(g - N0*N0*(zgrid(kk) - MinZ) - (*tracer[0])(ii,jj,kk));
            BPE = 0.0;

            // Compute energy diagnostics
            del_b   = pvmax(*tracer[0]) - pvmin(*tracer[0]);
            avg_rho = 1000.; //pssum(sum(*e_pv))/(size_x()*size_y()*size_z());
            KE      = 0.5*avg_rho*pssum(sum( u*u + v*v + w*w ))*Vol();
            PE      = 0.5*(avg_rho/(N0*N0))*pssum(sum(((*tracer[0])*(*tracer[0]))))*Vol();
            APE     = PE - BPE;

            // Determine if this is a plot interval
            if ((sim_time - nextplot) > -1e-5*plot_interval) {
                plot_now = true;
            }
            else { plot_now = false; }

            // Compute epv if we're going to need it
            if (plot_now) {
                ertel_pv(u, v, w, *tracer[0], e_pv, rot_f, N0,
                        length_x(), length_y(), length_z(),
                        size_x(),   size_y(),   size_z(),
                        type_x(),   type_y(),   type_z());
            }

            // Write chain outputs
            write_chains(w,tracer);
            chain_write_count++;

            // If it's time to output, then output
            if (plot_now) {
                
                // If appropriate, write 2D slices
                if (plot_write_ratio > 1) {
                    write_slices(u,v,w,tracer,e_pv);
                    slice_write++;
                    slice_write_count++;
                }

                // If appropraite, write 3D slices and associated information
                if (slice_write % plot_write_ratio == 0) {
                    plotnum = plotnum + 1;
                    t_step = MPI_Wtime(); // time just before write (for dump)
                    write_array(u,"u",plotnum);
                    write_array(v,"v",plotnum);
                    write_array(w,"w",plotnum);
                    write_array(*tracer[0],"b",plotnum);
                    write_array(*e_pv,"epv",plotnum);

                    // In order to be able to gracefully (and readily) restart,
                    // stitch the chain and slice files together now
                    stitch_chains(w);
                    stitch_slices(w);
                    prev_chain_write_count += chain_write_count;
                    prev_slice_write_count += slice_write_count;
                    chain_write_count = 0;
                    slice_write_count = 0;
                    reopen_chains();
                    reopen_slices();

                    clock_time = MPI_Wtime(); // time just afer write (for dump)
                    avg_write_time = (avg_write_time*(plotnum-restart_sequence-1) + (clock_time - t_step))/
                        (plotnum-restart_sequence);

                    // Find average time to write (for dump)
                    if (master()){
                        fprintf(stdout,"Last write time: %.6g. Average write time: %.6g.\n", clock_time - t_step, avg_write_time);
                    }

                    lastplot = itercount;
                    if (master()) {
                        plottimes = fopen("plot_times.txt","a");
                        assert(plottimes);
                        fprintf(plottimes,"%.10g\n",sim_time);
                        fclose(plottimes);
                        fprintf(stdout,"*");
                    }
                }
                nextplot = nextplot + plot_interval;

                if (sim_time - final_time > 1e-9) {
                    close_chains();
                    close_slices();
                }

            } 

            // Compute norms
            if (compute_norms) {
                b_pert_norm = pow(pssum(sum((*tracer[0] - *bg_b)*(*tracer[0] - *bg_b)))/(Nx*Ny*Nz),0.5); 
            }
            norm_u = pow(pssum(sum(u*u)),0.5);
            norm_v = pow(pssum(sum(v*v)),0.5);
            norm_w = pow(pssum(sum(w*w)),0.5);

            // Write a summary of the step 
            write_diagnostics(sim_time);
            last_writeout = itercount;

            // See if close to end of compute time and dump if necessary
            check_and_dump(clock_time, real_start_time, compute_time, sim_time, avg_write_time,
                    plotnum, u, v, w, tracer);

            // Change dump log file if successfully reached final time
            // the dump time will be twice final time so that a restart won't actually start
            successful_dump(plotnum, final_time, plot_interval);
        }

        // User-specified variables to dump
        void write_variables(DTArray & u,DTArray & v, DTArray & w,
                vector<DTArray *> & tracer) {
            write_array(u,"u.dump");
            write_array(v,"v.dump");
            write_array(w,"w.dump");
            write_array(*tracer[0],"b.dump");
            stitch_chains(w);
            stitch_slices(w);
            close_chains();
            close_slices();
        }


        void init_vels(DTArray & u, DTArray & v, DTArray & w) {
            // Initialize the velocities from read-in data
            if (master()) fprintf(stderr,"Initializing velocities\n");
            if (restarting and (!restart_from_dump)) {
                init_vels_restart(u, v, w);
            }
            else if (restarting and restart_from_dump) {
                init_vels_dump(u, v, w);
            }
            else {
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
                            fprintf(stdout,"Reading v from %s\n",
                                    v_filename.c_str());
                        read_array(v,v_filename.c_str(),Nx,Ny,Nz);
                        if (master()) 
                            fprintf(stdout,"Reading w from %s\n",
                                w_filename.c_str());
                        read_array(w,w_filename.c_str(),Nx,Ny,Nz);

                        break;
                }
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
            write_array(v,"v",plotnum);
            write_array(w,"w",plotnum);

        }

        void init_tracer(int t_num, DTArray & the_tracer) {
            if (master()) fprintf(stderr,"Initializing tracer %d\n",t_num);
            /* Initialize the density and take the opportunity to write out the grid */
            if (t_num == 0) {
                if (restarting and (!restart_from_dump)) {
                    init_tracer_restart("b",the_tracer);
                }
                else if (restarting and restart_from_dump) {
                    init_tracer_dump("b",the_tracer);
                }
                else {
                    switch (input_data_types) {
                        case MATLAB:
                            if (master())
                                fprintf(stderr,"reading matlab-type rho (%d x %d) from %s\n",
                                        Nx,Nz,rho_filename.c_str());
                            read_2d_slice(the_tracer,rho_filename.c_str(),Nx,Nz);
                            break;
                        case CTYPE:
                            if (master())
                                fprintf(stderr,"reading ctype rho (%d x %d) from %s\n",
                                        Nx,Nz,rho_filename.c_str());
                            read_2d_restart(the_tracer,rho_filename.c_str(),Nx,Nz);
                            break;
                        case FULL3D:
                            if (master())
                                fprintf(stderr,"reading rho (%d x %d x %d) from %s\n",
                                        Nx,Ny,Nz,rho_filename.c_str());
                            read_array(the_tracer,rho_filename.c_str(),Nx,Ny,Nz);
                            break;
                    }
                }
                write_array(the_tracer,"b",plotnum);
            } else if (t_num == 1) {
                if (restarting and (!restart_from_dump)) {
                    init_tracer_restart("dye1",the_tracer);
                }
                else if (restarting and restart_from_dump) {
                    init_tracer_dump("dye1",the_tracer);
                }
                else {
                    switch (input_data_types) {
                        case MATLAB:
                            if (master())
                                fprintf(stderr,"reading matlab-type tracer (%d x %d) from %s\n",
                                        Nx,Nz,tracer_filename.c_str());
                            read_2d_slice(the_tracer,tracer_filename.c_str(),Nx,Nz);
                            break;
                        case CTYPE:
                            if (master())
                                fprintf(stderr,"reading ctype tracer (%d x %d) from %s\n",
                                        Nx,Nz,tracer_filename.c_str());
                            read_2d_restart(the_tracer,tracer_filename.c_str(),Nx,Nz);
                            break;
                        case FULL3D:
                            if (master())
                                fprintf(stderr,"reading tracer (%d x %d x %d) from %s\n",
                                        Nx,Ny,Nz,tracer_filename.c_str());
                            //read_array(the_tracer,tracer_filename.c_str(),Nx,Ny,Nz);
                            break;
                    }
                }
            }
        }

        void forcing(double t, DTArray & u, DTArray & u_f,
                DTArray & v, DTArray & v_f, DTArray & w, DTArray & w_f,
                vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
            /* Velocity forcing */
            u_f = rot_f * v; 
            v_f = -rot_f * u;
            w_f = *(tracers[0]);
            // This is now b = bbar + b'. Vert forcing might change ...
            *(tracers_f[0]) = -N0*N0*w;
            /* if (tracer) {
             *  *(tracers_f[1]) = 0;
             *  w_f = w_f - tracer_g*((*tracers[1]));
             }*/
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
            Array<double,1> xgrid(split_range(Nx)), ygrid(Ny), zgrid(Nz);
            automatic_grid(MinX,MinY,MinZ, &xgrid, &ygrid, &zgrid);
        }
    }
};

int main(int argc, char ** argv) {
    MPI_Init(&argc,&argv);

    real_start_time = MPI_Wtime();     // for dump
    // To properly handle the variety of options, set up the boost
    // program_options library using the abbreviated interface in
    // ../Options.hpp

    options_init(); // Initialize options

    /* Grid options */
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

    /* Options for Grid Mapping */
    option_category("Grid mapping options");
    add_option("mapped_grid",&mapped,false,"Use a mapped (2D) grid");
    add_option("xgrid",&xgrid_filename,"x-grid filename");
    add_option("ygrid",&ygrid_filename,"","y-grid filename");
    add_option("zgrid",&zgrid_filename,"z-grid filename");

    /* Options for Input Data */
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

    /* Options for Second Tracer */
    option_category("Second tracer");
    add_switch("enable_tracer",&tracer,"Enable evolution of a second tracer");
    add_option("tracer_file",&tracer_filename,"Tracer filename");
    add_option("tracer_kappa",&tracer_kappa,"Diffusivity of tracer");
    add_option("tracer_gravity",&tracer_g,0.0,"Gravity for the second tracer");

    /* Physcial Parameters */
    option_category("Physical parameters");
    add_option("g",&g,9.81,"Gravitational acceleration");
    add_option("rot_f",&rot_f,0.0,"Coriolis force term");
    add_option("N0",&N0,0.0,"Buoyancy frequency term");
    add_option("visc",&vel_mu,0.0,"Kinematic viscosity");
    add_option("kappa",&dens_kappa,0.0,"Thermal diffusivity");
    add_option("perturbation",&perturb,0.0,"Veloc\tity perturbation (multiplicative white noise) applied to read-in data.");

    /* Running Options */
    option_category("Running options");
    add_option("init_time",&initial_time,0.0,"Initial time");
    add_option("final_time",&final_time,"Final time");
    add_option("plot_interval",&plot_interval,"Interval between output times");
    add_option("plot_write_ratio",&plot_write_ratio,1,"Ratio between plotting and writing");

    /* Restart Options */
    option_category("Restart options");
    add_switch("restart",&restarting,"Restart from prior output time. OVERRIDES many other values.");
    add_option("restart_time",&restart_time,0.0,"Time to restart from");
    add_option("restart_sequence",&restart_sequence,
            "Sequence number to restart from (if plot_interval has changed)");

    /* Filtering Options */
    option_category("Filtering options");
    add_option("f_strength",&f_strength,20.0,"filter strength");
    add_option("f_cutoff",&f_cutoff,0.6,"filter cutoff");
    add_option("f_order",&f_order,2.0,"fiter order");

    /* Diagnostics Options */
    option_category("Diagnostics options");
    add_option("compute_norms",&compute_norms,false,"Compute epv perturbation norms?");
    add_option("bg_b_filename",&bg_b_filename,"","Where to find epv basic state if needed.");

    option_category("Dumping options");
    add_option("compute_time",&compute_time,-1.0,"Time permitted for computation");
    add_option("restart_from_dump",&restart_from_dump,false,"If restart from dump");

    // Parse the options from the command line and config file
    options_parse(argc,argv);
    max_dt = (2*M_PI/N0)/20.0;

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
