#include "BaseCase.hpp"
#include "Science.hpp"
#include "NSIntegrator.hpp"
#include "TArray.hpp"
#include <blitz/array.h>
#include <math.h>

//using namespace TArray;
using namespace NSIntegrator;
using blitz::Array;
using std::vector;

/* Call the source code writing function in the constructor */
extern "C" {
    void WriteCaseFileSource(void);
}
BaseCase::BaseCase(void)
{
    if (master()) WriteCaseFileSource();
}

/* Implementation of non-abstract methods in BaseCase */
int BaseCase::numActive() const { return 0; }
int BaseCase::numPassive() const { return 0; }
int BaseCase::numtracers() const { /* total number of tracers */
    return numActive() + numPassive();
}

int BaseCase::size_x() const {
    return size_cube();
}
int BaseCase::size_y() const {
    return size_cube();
}
int BaseCase::size_z() const {
    return size_cube();
}
double BaseCase::length_x() const {
    return length_cube();
}
double BaseCase::length_y() const {
    return length_cube();
}
double BaseCase::length_z() const {
    return length_cube();
}

DIMTYPE BaseCase::type_x() const {
    return type_default();
}
DIMTYPE BaseCase::type_y() const {
    return type_default();
}
DIMTYPE BaseCase::type_z() const {
    return type_default();
}
DIMTYPE BaseCase::type_default() const {
    return PERIODIC;
}

void BaseCase::tracer_bc_x(int t_num, double & dir, double & neu) const {
    if (!zero_tracer_boundary) {
        dir = 0; 
        neu = 1;
    }
    else {
        dir = 1;
        neu = 0;
    }
    return;
}
void BaseCase::tracer_bc_y(int t_num, double & dir, double & neu) const {
    if (!zero_tracer_boundary) {
        dir = 0; 
        neu = 1;
    }
    else {
        dir = 1;
        neu = 0;
    }
    return;
}
void BaseCase::tracer_bc_z(int t_num, double & dir, double & neu) const {
    if (!zero_tracer_boundary) {
        dir = 0; 
        neu = 1;
    }
    else {
        dir = 1;
        neu = 0;
    }
    return;
}
bool BaseCase::tracer_bc_forcing() const {
    return false;
}
bool BaseCase::is_mapped() const { // Whether this problem has mapped coordinates
    return false;
}
// Coordinate mapping proper, if is_mapped() returns true.  This features full,
// 3D arrays, but at least initially we're restricting ourselves to 2D (x,z)
// mappings
void BaseCase::do_mapping(DTArray & xgrid, DTArray & ygrid, DTArray & zgrid) {
    return;
}

/* Physical parameters */
double BaseCase::get_visco()            const { return 0; }
double BaseCase::get_diffusivity(int t) const { return 0; }
double BaseCase::get_rot_f()            const { return 0; }
int BaseCase::get_restart_sequence()    const { return 0; }
double BaseCase::get_plot_interval()    const { return 0; }
double BaseCase::get_dt_max()           const { return 0; }
double BaseCase::get_next_plot()              { return 0; }

/* Initialization */
double BaseCase::init_time() const {
    return 0;
}
void BaseCase::init_tracers(vector<DTArray *> & tracers) {
    /* Initalize tracers one-by-one */
    if (numtracers() == 0) return; // No tracers, do nothing
    assert(numtracers() == int(tracers.size())); // Sanity check
    for (int i = 0; i < numtracers(); i++) {
        init_tracer(i, *(tracers[i]));
    }
}

/* Check and modify the timestep in order to land evenly on a plot time */
double BaseCase::check_timestep(double step, double now) {
    // Check time step
    if (step < 1e-9) {
        // Timestep's too small, somehow stuff is blowing up
        if (master()) fprintf(stderr,"Tiny timestep (%e), aborting\n",step);
        return -1;
    } else if (step > get_dt_max()) {
        // Cap the maximum timestep size
        step = get_dt_max();
    }

    // Now, calculate how many timesteps remain until the next writeout
    double until_plot = get_next_plot() - now;
    int steps = ceil(until_plot / step);
    // Where will we be after (steps) timesteps of the current size?
    double real_until_plot = steps*step;
    // If that's close enough to the real writeout time, that's fine.
    if (fabs(until_plot - real_until_plot) < 1e-6*get_plot_interval()) {
        return step;
    } else {
        // Otherwise, square up the timeteps.  This will always shrink the timestep.
        return (until_plot / steps);
    }
}

/* Forcing */
void BaseCase::forcing(double t,  DTArray & u, DTArray & u_f,
        DTArray & v, DTArray & v_f,  DTArray & w, DTArray & w_f,
        vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
    /* First, if no active tracers then use the simpler form */
    if (numActive() == 0) {
        passive_forcing(t, u, u_f, v, v_f, w, w_f);
    } else {
        /* Look at split velocity-tracer forcing */
        vel_forcing(t, u_f, v_f, w_f, tracers);
        tracer_forcing(t, u, v, w, tracers_f);
    }
    /* And any/all passive tracers get 0 forcing */
    for (int i = 0; i < numPassive(); i++) {
        *(tracers_f[numActive() + i]) = 0;
    }
}
void BaseCase::passive_forcing(double t,  DTArray & u, DTArray & u_f,
        DTArray & v, DTArray & v_f,  DTArray &, DTArray & w_f) {
    /* Reduce to velocity-independent case */
    stationary_forcing(t, u_f, v_f, w_f);
}
void BaseCase::stationary_forcing(double t, DTArray & u_f, DTArray & v_f, 
        DTArray & w_f) {
    /* Default case, 0 forcing */
    u_f = 0;
    v_f = 0;
    w_f = 0;
}

/* Analysis */
void BaseCase::analysis(double t, DTArray & u, DTArray & v, DTArray & w,
        vector<DTArray *> tracer, DTArray & pres) {
    analysis(t,u,v,w,tracer);
}
void BaseCase::analysis(double t, DTArray & u, DTArray & v, DTArray & w,
        vector<DTArray *> tracer) {
    /* Do velocity and tracer analysis seperately */
    vel_analysis(t, u, v, w);
    for (int i = 0; i < numtracers(); i++) {
        tracer_analysis(t, i, *(tracer[i]));
    } 
}

void BaseCase::automatic_grid(double MinX, double MinY, double MinZ,
        Array<double,1> * xx, Array<double,1> * yy, Array<double,1> * zz){
    //Array<double,1> xx(split_range(size_x())), yy(size_y()), zz(size_z());
    bool xxa = false, yya = false, zza = false;
    if (!xx) {
        xxa = true; // Delete xx when returning
        xx = new Array<double,1>(split_range(size_x()));
    }
    if (!yy) {
        yya = true;
        yy = new Array<double,1>(size_y());
    }
    if (!zz) {
        zza = true;
        zz = new Array<double,1>(size_z());
    }
    Array<double,3> grid(alloc_lbound(size_x(),size_y(),size_z()),
            alloc_extent(size_x(),size_y(),size_z()),
            alloc_storage(size_x(),size_y(),size_z()));
    blitz::firstIndex ii;
    blitz::secondIndex jj;
    blitz::thirdIndex kk;

    // Generate 1D arrays
    if (type_x() == NO_SLIP) {
        *xx = MinX+length_x()*(0.5-0.5*cos(M_PI*ii/(size_x()-1)));
    } else {
        *xx = MinX + length_x()*(ii+0.5)/size_x();
    }
    *yy = MinY + length_y()*(ii+0.5)/size_y();
    if (type_z() == NO_SLIP) {
        *zz = MinZ+length_z()*(0.5-0.5*cos(M_PI*ii/(size_z()-1)));
    } else {
        *zz = MinZ + length_z()*(0.5+ii)/size_z();
    }

    // Write grid/reader
    grid = (*xx)(ii) + 0*jj + 0*kk;
    write_array(grid,"xgrid");
    write_reader(grid,"xgrid",false);

    if (size_y() > 1) {
        grid = 0*ii + (*yy)(jj) + 0*kk;
        write_array(grid,"ygrid");
        write_reader(grid,"ygrid",false);
    }

    grid = 0*ii + 0*jj + (*zz)(kk);
    write_array(grid,"zgrid");
    write_reader(grid,"zgrid",false);

    // Clean up
    if (xxa) delete xx;
    if (yya) delete yy;
    if (zza) delete zz;
}

/* Read velocities from matlab output */
void BaseCase::init_vels_matlab(DTArray & u, DTArray & v, DTArray & w,
        const std::string & u_filename, const std::string & v_filename,
        const std::string & w_filename) {
    init_matlab("u",u_filename,u);
    if ( size_y()>1 or get_rot_f()!=0 ) {
        init_matlab("v",v_filename,v);
    } else {
        v = 0;
    }
    init_matlab("w",w_filename,w);
}

/* Read velocities from ctype output */
void BaseCase::init_vels_ctype(DTArray & u, DTArray & v, DTArray & w,
        const std::string & u_filename, const std::string & v_filename,
        const std::string & w_filename) {
    init_ctype("u",u_filename,u);
    if ( size_y()>1 or get_rot_f()!=0 ) {
        init_ctype("v",v_filename,v);
    } else {
        v = 0;
    }
    init_ctype("w",w_filename,w);
}

/* Read velocities from regular output */
void BaseCase::init_vels_restart(DTArray & u, DTArray & v, DTArray & w) {
    init_tracer_restart("u", u);
    if ( size_y()>1 or get_rot_f()!=0 ) {
        init_tracer_restart("v", v);
    } else {
        v = 0;
    }
    init_tracer_restart("w", w);
}

/* Read velocities from dump output */
void BaseCase::init_vels_dump(DTArray & u, DTArray & v, DTArray & w){
    if (master()) fprintf(stdout,"Reading u from u.dump\n");
    read_array_par(u,"u.dump",size_x(),size_y(),size_z());
    if ( size_y()>1 or get_rot_f()!=0 ) {
        if (master()) fprintf(stdout,"Reading v from v.dump\n");
        read_array_par(v,"v.dump",size_x(),size_y(),size_z());
    }
    if (master()) fprintf(stdout,"Reading w from w.dump\n");
    read_array_par(w,"w.dump",size_x(),size_y(),size_z());
    return;
}

/* Read grid from regular output */
void BaseCase::init_grid_restart(const std::string & component,
        const std::string & filename, DTArray & grid){
    if (master()) fprintf(stdout,"Reading %s from %s\n",component.c_str(),filename.c_str());
    read_array_par(grid,filename.c_str(),size_x(),size_y(),size_z());
    return;
}

/* Read field from regular output */
void BaseCase::init_tracer_restart(const std::string & field, DTArray & the_tracer){
    char filename[100];
    snprintf(filename,100,"%s.%d",field.c_str(),get_restart_sequence());
    if (master()) fprintf(stdout,"Reading %s from %s\n",field.c_str(),filename);
    read_array_par(the_tracer,filename,size_x(),size_y(),size_z());
    return;
}

/* Read field from dump output */
void BaseCase::init_tracer_dump(const std::string & field, DTArray & the_tracer){
    char filename[100];
    snprintf(filename,100,"%s.dump",field.c_str());
    if (master()) fprintf(stdout,"Reading %s from %s\n",field.c_str(),filename);
    read_array_par(the_tracer,filename,size_x(),size_y(),size_z());
    return;
}

/* Read field from matlab data */
void BaseCase::init_matlab(const std::string & field,
        const std::string & filename, DTArray & the_field){
    if (master()) fprintf(stdout,"Reading MATLAB-format %s (%d x %d) from %s\n",
            field.c_str(),size_x(),size_z(),filename.c_str());
    read_2d_slice(the_field,filename.c_str(),size_x(),size_z());
    return;
}

/* Read field from CTYPE data */
void BaseCase::init_ctype(const std::string & field,
        const std::string & filename, DTArray & the_field){
    if (master()) fprintf(stdout,"Reading CTYPE-format %s (%d x %d) from %s\n",
            field.c_str(),size_x(),size_z(),filename.c_str());
    read_2d_restart(the_field,filename.c_str(),size_x(),size_z());
    return;
}

/* write out vertical chain of data */
void BaseCase::write_chain(const char *filename, DTArray & val, int Iout, int Jout, double time) {
    FILE *fid=fopen(filename,"a");
    if (fid == NULL) {
        fprintf(stderr,"Unable to open %s for writing\n",filename);
        MPI_Finalize(); exit(1);
    }
    fprintf(fid,"%g",time);
    for (int ki=0; ki<size_z(); ki++) fprintf(fid," %g",val(Iout,Jout,ki));
    fprintf(fid,"\n");
    fclose(fid);
}

/* Check and dump */
void BaseCase::check_and_dump(double clock_time, double real_start_time,
        double compute_time, double sim_time, double avg_write_time, int plot_number, int itercount,
        DTArray & u, DTArray & v, DTArray & w, vector<DTArray *> & tracer){
    int do_dump = 0;
    if (master()) {
        double total_run_time = clock_time - real_start_time;
        double needed_time = 10*avg_write_time + 2*total_run_time/itercount;

        // check if close to end of compute time
        if (compute_time>0 and (compute_time - total_run_time < needed_time)){
            do_dump = 1; // true
        }
    }
    MPI_Bcast(&do_dump, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (do_dump == 1) {
        if (master()){
            fprintf(stdout,"Too close to final time, dumping!\n");
        }
        write_variables(u, v, w, tracer);

        // Write the dump time to a text file for restart purposes
        if (master()){
            FILE * dump_file; 
            dump_file = fopen("dump_time.txt","w");
            assert(dump_file);
            fprintf(dump_file,"The dump time was:\n%.12g\n", sim_time);
            fprintf(dump_file,"The dump index was:\n%d\n", plot_number);
            fclose(dump_file);
        }

        // Die gracefully
        MPI_Finalize(); exit(0);
    }
}

/* Change dump log file for successful completion */
void BaseCase::successful_dump(int plot_number, double final_time, double plot_interval) {
    if (master() and (plot_number == final_time/plot_interval)){
        // Write the dump time to a text file for restart purposes
        FILE * dump_file; 
        dump_file = fopen("dump_time.txt","w");
        assert(dump_file);
        fprintf(dump_file,"The dump 'time' was:\n%.12g\n", 2*final_time);
        fprintf(dump_file,"The dump index was:\n%d\n", plot_number);
        fclose(dump_file);
    }
}

/* Write plot time information */
void BaseCase::write_plot_times(double write_time, double avg_write_time,
        double plot_interval, int plotnum, bool restarting, double time) {
    if (master()) {
        // in log file
        fprintf(stdout,"Write time: %.6g. Average write time: %.6g.\n",
                write_time, avg_write_time);
        fprintf(stdout,"*");
        // track in specific file
        FILE * plottimes_file = fopen("plot_times.txt","a");
        assert(plottimes_file);
        if ( plotnum==get_restart_sequence()+1 and !restarting )
            fprintf(plottimes_file,"Output number, Simulation time (s), "
                    "Write time (s), Average write time (s)\n");
        fprintf(plottimes_file,"%d, %.12f, %.12g, %.12g\n",
                plotnum, time, write_time, avg_write_time);
        fclose(plottimes_file);
    }
}

void BaseCase::stresses(TArrayn::DTArray & u, TArrayn::DTArray & v, TArrayn::DTArray & w,
        TArrayn::DTArray & Hprime, TArrayn::DTArray & temp, TArrayn::Grad * gradient_op,
        const string * grid_type, const double mu, double time, int itercount, bool restarting) {
    // set-up
    static DTArray *tx = alloc_array(size_x(),size_y(),1);
    static DTArray *ty = alloc_array(size_x(),size_y(),1);
    blitz::firstIndex ii;
    blitz::secondIndex jj;

    // bottom stress ( along topography - x )
    bottom_stress_x(*tx, Hprime, u, w, temp, gradient_op, grid_type, size_z(), mu);
    double bot_tx_tot = pssum(sum(
                (*get_quad_x())(ii)*pow(1+pow(Hprime,2),0.5)*
                (*get_quad_y())(jj)*(*tx)));
    double bot_tx_abs = pssum(sum(
                (*get_quad_x())(ii)*pow(1+pow(Hprime,2),0.5)*
                (*get_quad_y())(jj)*abs(*tx)));
    // bottom stress ( across topography - y )
    bottom_stress_y(*ty, Hprime, v, temp, gradient_op, grid_type, size_z(), mu);
    double bot_ty_tot = pssum(sum(
                (*get_quad_x())(ii)*pow(1+pow(Hprime,2),0.5)*
                (*get_quad_y())(jj)*(*ty)));
    double bot_ty_abs = pssum(sum(
                (*get_quad_x())(ii)*pow(1+pow(Hprime,2),0.5)*
                (*get_quad_y())(jj)*abs(*ty)));
    double bot_ts = pssum(sum(
                (*get_quad_x())(ii)*pow(1+pow(Hprime,2),0.5)*
                (*get_quad_y())(jj)*pow(pow(*tx,2)+pow(*ty,2),0.5)));
    // top stress ( along "topography" - x )
    top_stress_x(*tx, u, temp, gradient_op, grid_type, mu);
    double top_tx_tot = pssum(sum(
                (*get_quad_x())(ii)*
                (*get_quad_y())(jj)*(*tx)));
    double top_tx_abs = pssum(sum(
                (*get_quad_x())(ii)*
                (*get_quad_y())(jj)*abs(*tx)));
    // top stress ( across "topography" - y )
    top_stress_y(*ty, v, temp, gradient_op, grid_type, mu);
    double top_ty_tot = pssum(sum(
                (*get_quad_x())(ii)*
                (*get_quad_y())(jj)*(*ty)));
    double top_ty_abs = pssum(sum(
                (*get_quad_x())(ii)*
                (*get_quad_y())(jj)*abs(*ty)));
    double top_ts = pssum(sum(
                (*get_quad_x())(ii)*
                (*get_quad_y())(jj)*pow(pow(*tx,2)+pow(*ty,2),0.5)));
    // add to a stress diagnostic file
    if (master()) {
        FILE * stresses_file = fopen("stresses.txt","a");
        assert(stresses_file);
        if ( itercount==1 and !restarting )
            fprintf(stresses_file,"Time, "
                    "Bottom_tx_tot, Bottom_tx_abs, Bottom_ty_tot, Bottom_ty_abs, Bottom_ts, "
                    "Top_tx_tot, Top_tx_abs, Top_ty_tot, Top_ty_abs, Top_ts\n");
        fprintf(stresses_file,"%.12f, "
                "%.12g, %.12g, %.12g, %.12g, %.12g, "
                "%.12g, %.12g, %.12g, %.12g, %.12g\n",
                time,
                bot_tx_tot, bot_tx_abs, bot_ty_tot, bot_ty_abs, bot_ts,
                top_tx_tot, top_tx_abs, top_ty_tot, top_ty_abs, top_ts);
        fclose(stresses_file);
    }
}

// Enstrophy Density: 1/2*(vort_x^2 + vort_y^2 + vort_z^2)
void BaseCase::enstrophy(TArrayn::DTArray & u, TArrayn::DTArray & v, TArrayn::DTArray & w,
        TArrayn::DTArray & temp, TArrayn::Grad * gradient_op, const string * grid_type,
        double time, int itercount, bool restarting) {
    // set-up
    blitz::firstIndex ii;
    blitz::secondIndex jj;
    blitz::thirdIndex kk;
    // compute components
    compute_vort_x(v, w, temp, gradient_op, grid_type);
    double enst_x_tot = pssum(sum(0.5*pow(temp,2)*
                (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
    compute_vort_y(u, w, temp, gradient_op, grid_type);
    double enst_y_tot = pssum(sum(0.5*pow(temp,2)*
                (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
    compute_vort_z(u, v, temp, gradient_op, grid_type);
    double enst_z_tot = pssum(sum(0.5*pow(temp,2)*
                (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk)));
    double enst_tot = enst_x_tot + enst_y_tot + enst_z_tot;
    // add to a enstrophy diagnostic file
    if (master()) {
        FILE * enst_file = fopen("enstrophy.txt","a");
        assert(enst_file);
        if ( itercount==1 and !restarting )
            fprintf(enst_file,"Time, enst_x, enst_y, enst_z, enst_tot\n");
        fprintf(enst_file,"%.12f, %.12g, %.12g, %.12g, %.12g\n",
                time, enst_x_tot, enst_y_tot,enst_z_tot, enst_tot);
        fclose(enst_file);
    }
}
