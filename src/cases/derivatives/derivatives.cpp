/* Derivative case file for computing derivatives of existing fields */
// only able to return first and second derivatives

/* ------------------ Top matter --------------------- */

// Required headers
#include "../BaseCase.hpp"      // contains default class
#include "../Options.hpp"       // config-file parser

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
// Expansion types and indices
S_EXP expan[3];
const int x_ind = 0;
const int y_ind = 1;
const int z_ind = 2;

// physical parameters
double visco;                       // viscosity (m^2/s)
double rho_0;                       // reference density (kg/m^3)

// Derivative options
string deriv_filenames;              // file name to take derivative of
int start_sequence;                 // output number to start taking derivatives at
int final_sequence;                 // output number to stop  taking derivatives at
int step_sequence;                  // step between outputs to take derivatives
bool deriv_x, deriv_y, deriv_z;     // which derivatives
bool do_vor_x, do_vor_y, do_vor_z;  // Do vorticity calculations?
bool do_enstrophy;                  // Do Enstrophy calculation?
bool do_dissipation;                // Do Viscous dissipation?
bool v_exist;                       // Does the v field exist?

/* ------------------ Adjust the class --------------------- */

class userControl : public BaseCase {
    public:
        /* Initialize things */
        Grad * gradient_op;     // gradient operator
        DTArray deriv_var;      // array for derivative

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

        /* Set other things */
        double get_visco() const { return visco; }

        /* Read grid (if mapped) */
        bool is_mapped() const { return mapped; }
        void do_mapping(DTArray & xg, DTArray & yg, DTArray & zg) {
            init_grid_restart("x","xgrid",xg);
            if ( Ny > 1 )
                init_grid_restart("y","ygrid",yg);
            init_grid_restart("z","zgrid",zg);
        }

        /* function for splitting a string - to parse deriv_filenames */
        void split(const string &s, char delim, vector<string> &elems) {
            stringstream ss(s);
            string item;
            while (getline(ss, item, delim)) {
                elems.push_back(item);
            }
        }

        /* Read fields and do derivatives */
        void init_vels(DTArray & u, DTArray & v, DTArray & w) {
            // set-up
            assert(gradient_op);
            char filename[100];
            bool saved_v = false;
            bool saved_w = false;
            string prev_deriv, base_field;
            vector<string> fields;      // vector of fields to take derivatives
            split(deriv_filenames.c_str(), ' ', fields);    // populate that vector

            // Compute derivatives at each requested output
            for ( int plotnum = start_sequence; plotnum <= final_sequence;
                    plotnum = plotnum + step_sequence ) {
                if ( deriv_x or deriv_y or deriv_z ) {
                    // loop over each field
                    for ( int var_num = 0; var_num <= fields.size()-1; var_num++ ) {
                        // check if field is a derivative field
                        bool input_deriv = false;        // assume it's not a derivative
                        int var_len = fields[var_num].length();
                        if ( var_len > 2 ) {
                            if ( fields[var_num].substr(var_len-2,1) == "_" ) {
                                // if second last char is an underscore then its a derivative field
                                input_deriv = true;
                                prev_deriv = fields[var_num].substr(var_len-1,1);  // the completed derivative
                                base_field = fields[var_num].substr(0,var_len-2);  // the differentiated field
                            }
                        }

                        // parse for expansion type
                        find_expansion(grid_type, expan, fields[var_num], base_field);
                        // adjust for a second derivative
                        if ( input_deriv == true ) {
                            if      ( prev_deriv == "x" ) { expan[x_ind] = swap_trig(expan[x_ind]); }
                            else if ( prev_deriv == "y" ) { expan[y_ind] = swap_trig(expan[y_ind]); }
                            else if ( prev_deriv == "z" ) { expan[z_ind] = swap_trig(expan[z_ind]); }
                        }

                        // read the field and setup for derivative 
                        if ( fields[var_num] == "v" ) {
                            saved_v = true;
                            init_tracer_plotnum(fields[var_num],v,plotnum);
                            gradient_op->setup_array(&v,expan[x_ind],expan[y_ind],expan[z_ind]); }
                        else if ( fields[var_num] == "w" ) {
                            saved_w = true;
                            init_tracer_plotnum(fields[var_num],w,plotnum);
                            gradient_op->setup_array(&w,expan[x_ind],expan[y_ind],expan[z_ind]); }
                        else {
                            // else use u to hold the field
                            init_tracer_plotnum(fields[var_num],u,plotnum);
                            gradient_op->setup_array(&u,expan[x_ind],expan[y_ind],expan[z_ind]);
                        }
                        if (master()) {
                            fprintf(stdout,"Expansions are (x,y,z): (%s, %s, %s)\n",
                                    S_EXP_NAME[expan[x_ind]],S_EXP_NAME[expan[y_ind]],
                                    S_EXP_NAME[expan[z_ind]]);
                        }

                        // X derivative
                        if (deriv_x) {
                            gradient_op->get_dx(&deriv_var,false);
                            double max_var = psmax(max(abs(deriv_var)));
                            if (master()) fprintf(stdout,"Max x derivative: %.6g\n",max_var);

                            // save the derivative
                            if ( !input_deriv ) {
                                snprintf(filename,100,"%s_x",fields[var_num].c_str()); }
                            else {
                                snprintf(filename,100,"%sx",fields[var_num].c_str());
                            }
                            write_array(deriv_var,filename,plotnum);
                        }
                        // Y derivative
                        if (deriv_y) {
                            gradient_op->get_dy(&deriv_var,false);
                            double max_var = psmax(max(abs(deriv_var)));
                            if (master()) fprintf(stdout,"Max y derivative: %.6g\n",max_var);

                            // save the derivative
                            if ( !input_deriv ) {
                                snprintf(filename,100,"%s_y",fields[var_num].c_str()); }
                            else {
                                snprintf(filename,100,"%sy",fields[var_num].c_str());
                            }
                            write_array(deriv_var,filename,plotnum);
                        }
                        // Z derivative
                        if (deriv_z) {
                            gradient_op->get_dz(&deriv_var,false);
                            double max_var = psmax(max(abs(deriv_var)));
                            if (master()) fprintf(stdout,"Max z derivative: %.6g\n",max_var);

                            // save the derivative
                            if ( !input_deriv ) {
                                snprintf(filename,100,"%s_z",fields[var_num].c_str()); }
                            else {
                                snprintf(filename,100,"%sz",fields[var_num].c_str());
                            }
                            write_array(deriv_var,filename,plotnum);
                        }
                    }
                }

                // read in fields (if not already stored in memory)
                if ( do_vor_x or do_vor_y or do_vor_z or do_enstrophy or do_dissipation) {
                    // u
                    init_tracer_plotnum("u",u,plotnum);
                    // v
                    if ( !saved_v ) {
                        if ( v_exist ) {
                            init_tracer_plotnum("v",v,plotnum); }
                        else {
                            if (master()) fprintf(stdout,"No v field, setting v=0\n");
                            v = 0;
                        }
                    }
                    // w
                    if ( !saved_w ) {
                        init_tracer_plotnum("w",w,plotnum);
                    }
                }

                // X-component of vorticity
                if (do_vor_x) {
                    compute_vort_x(v, w, deriv_var, gradient_op, grid_type);
                    double max_var = psmax(max(abs(deriv_var)));
                    if (master()) fprintf(stdout,"Max X-vorticity: %.6g\n",max_var);
                    write_array(deriv_var,"vortx",plotnum);
                }
                // Y-component of vorticity
                if (do_vor_y) {
                    compute_vort_y(u, w, deriv_var, gradient_op, grid_type);
                    double max_var = psmax(max(abs(deriv_var)));
                    if (master()) fprintf(stdout,"Max Y-vorticity: %.6g\n",max_var);
                    write_array(deriv_var,"vorty",plotnum);
                }
                // Z-component of vorticity
                if (do_vor_z) {
                    compute_vort_z(u, v, deriv_var, gradient_op, grid_type);
                    double max_var = psmax(max(abs(deriv_var)));
                    if (master()) fprintf(stdout,"Max Z-vorticity: %.6g\n",max_var);
                    write_array(deriv_var,"vortz",plotnum);
                }

                // Calculate Enstrophy
                if ( do_enstrophy ) {
                    enstrophy_density(u, v, w, deriv_var, gradient_op, grid_type,
                            Nx, Ny, Nz);
                    double tot_enst = pssum(sum(
                                (*get_quad_x())(ii)*
                                (*get_quad_y())(jj)*
                                (*get_quad_z())(kk)*deriv_var));
                    if (master()) fprintf(stdout,"Total Enstrophy: %.6g\n",tot_enst);
                    write_array(deriv_var,"enst",plotnum);
                }
                // Calculate Viscous dissipation
                if ( do_dissipation ) {
                    double mu = visco*rho_0;    // dynamic viscosity
                    dissipation(u, v, w, deriv_var, gradient_op, grid_type,
                            Nx, Ny, Nz, mu);
                    double tot_diss = pssum(sum(
                                (*get_quad_x())(ii)*
                                (*get_quad_y())(jj)*
                                (*get_quad_z())(kk)*deriv_var));
                    if (master()) fprintf(stdout,"Total Dissipation: %.6g\n",tot_diss);
                    write_array(deriv_var,"diss",plotnum);
                }
            }
        }

        // Constructor: Initialize local variables
        userControl() :
            gradient_op(0),
            deriv_var(alloc_lbound(Nx,Ny,Nz),
                    alloc_extent(Nx,Ny,Nz),
                    alloc_storage(Nx,Ny,Nz))
    {   compute_quadweights(
            size_x(),   size_y(),   size_z(),
            length_x(), length_y(), length_z(),
            type_x(),   type_y(),   type_z());
    }
};

/* The ''main'' routine */
int main(int argc, char ** argv) {
    /* Initialize MPI.  This is required even for single-processor runs,
       since the inner routines assume some degree of parallelization,
       even if it is trivial. */
    MPI_Init(&argc,&argv);

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

    option_category("Grid mapping options");
    add_option("mapped_grid",&mapped,false,"Is the grid mapped?");

    string xgrid_type, ygrid_type, zgrid_type;
    add_option("type_x",&xgrid_type,
            "Grid type in X.  Valid values are:\n"
            "   FOURIER: Periodic\n"
            "   FREE_SLIP: Cosine expansion\n"
            "   NO_SLIP: Chebyhsev expansion");
    add_option("type_y",&ygrid_type,"FOURIER","Grid type in Y");
    add_option("type_z",&zgrid_type,"Grid type in Z");

    option_category("Physical parameters");
    add_option("visco",&visco,0.0,"Viscosity");
    add_option("rho_0",&rho_0,1.0,"Reference Density");

    option_category("Derivative options");
    add_option("deriv_files",&deriv_filenames,"Derivative filename");
    add_option("start_sequence",&start_sequence,"Sequence number to start taking derivatives at");
    add_option("final_sequence",&final_sequence,"Sequence number to stop  taking derivatives at");
    add_option("step_sequence",&step_sequence,1,"Step between outputs to take derivatives");
    add_option("deriv_x",&deriv_x,false,"Do the x derivative?");
    add_option("deriv_y",&deriv_y,false,"Do the y derivative?");
    add_option("deriv_z",&deriv_z,false,"Do the z derivative?");
    add_option("do_vor_x",&do_vor_x,false,"Do the X-component of vorticity?");
    add_option("do_vor_y",&do_vor_y,false,"Do the Y-component of vorticity?");
    add_option("do_vor_z",&do_vor_z,false,"Do the Z-component of vorticity?");
    add_option("do_enstrophy",&do_enstrophy,false,"Calculate enstrophy?");
    add_option("do_dissipation",&do_dissipation,false,"Calculate viscous dissipation?");
    add_option("v_exist",&v_exist,"Does the v field exist?");

    // Parse the options from the command line and config file
    options_parse(argc,argv);

    /* ------------------ Adjust and check parameters --------------------- */
    /* Now, make sense of the options received.  Many of these values
       can be directly used, but the ones of string-type need further procesing. */

    // parse expansion types
    get_expansions(xgrid_type, ygrid_type, zgrid_type, intype_x, intype_y, intype_z);
    // vector of string types
    grid_type[x_ind] = xgrid_type;
    grid_type[y_ind] = ygrid_type;
    grid_type[z_ind] = zgrid_type;

    // adjust Ly for 2D
    if (Ny==1 and Ly!=1.0){
        Ly = 1.0;
        if (master())
            fprintf(stdout,"Simulation is 2 dimensional, "
                    "Ly has been changed to 1.0 for normalization.\n");
    }

    /* ------------------ Do stuff --------------------- */
    userControl mycode; // Create an instantiated object of the above class
    FluidEvolve<userControl> kevin_kh(&mycode);
    kevin_kh.initialize();
    MPI_Finalize();
    return 0;
}
