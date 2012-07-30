#ifndef MULTIGRID_HPP
#define MULTIGRID_HPP 1

#include <mpi.h>
#include <blitz/array.h>
#include "umfpack.h"

/* Header file for our required 2D multigrid solver, operating on a (split) grid
   with the numerical Jacobian entries (from the spectral model) providing varying
   coefficients */
enum SYM_TYPE {
   SYM_NONE=0, // No symmetry
   SYM_PERIODIC, // Periodic
   SYM_EVEN, 
   SYM_ODD
} ;

enum CYCLE_TYPE { // Multigrid cycle type
   CYCLE_V, // down-up, no recursion
   CYCLE_W, // down-up-down-up with full recursion
   CYCLE_F, // down-up-down-up with recursion only on the first half
   CYCLE_NONE // Debugging -- relax only
};

class MG_Solver {
   public:
      /* Constructor.  Gets x and z values of the numerical (perfectly orthogonal)
         grid, for construction of the 1D finite difference operators.  These arrays
         must also be for the full domain, since boundary points are special.

         If symmetry_problem is true, then xvals is extended by two to allow for
         a ghost-point type handling of the left and right boundaries */
      MG_Solver(blitz::Array<double,1> xvals, blitz::Array<double,1> zvals,
            SYM_TYPE symmetry_problem, MPI_Comm comm = MPI_COMM_WORLD);

      /* Set the specific form of symmetry.  This is not in the constructor because
         it might change between invocations on the same grid.  As an example, on
         a rectanguar grid with free-slip conditions, u (orthogonal to the boundary)
         is odd symmetry while w is even.  Pressure likewise has even symmetry */
      void set_x_symmetry(SYM_TYPE type);

      /* Note that after this point all arrays specified are split by processor */
      
      /* Problem setup, which takes the several coefficients.  With the exception
         of DC, these will probably not change between invocations.  For
         the multilevel algorithm, this can call downwards recursively with
         coarsened versions of these operators */
      void problem_setup(blitz::Array<double,2> & Uxx, blitz::Array<double,2> & Uzz,
            blitz::Array<double,2> & Uxz, blitz::Array<double,2> & Ux,
            blitz::Array<double,2> & Uz);

      /* Setting the DC term.  This will change based on timestep, diffusivity,
         or spanwise wavenumber (in the case of a reduced 3D problem).  The
         general case has this varying, but I much prefer not to deal with that
         right now */
      void helmholtz_setup(double c);

      /* BC setup; so many terms are necessary because physical grid lines might not
         intersect the boundary at a right angle.  Indeed, they won't at the bottom
         even in the case of a hill.  To avoid a duplicate function spec, this
         fuction gets called twice with dim=0 for x and dim=1 for z*/
      void bc_setup(int dim,blitz::Array<double,1> u_min,
            blitz::Array<double,1> uz_min,blitz::Array<double,1> ux_min,
            blitz::Array<double,1> u_max, blitz::Array<double,1> uz_max,
            blitz::Array<double,1> ux_max);

      /* Perform a multigrid cycle, or at the coarse level do a direct solve
         via UMFPACK.  Takes as input the desired cycle type (F-cycle or V-cycle),
         and input residual f, and returns as output the (approximate) u.

         For the coarse level, this does the coarse solve regardless of cycle type.

         The bonus_in and bonus_out terms are required for indefinite problems, where
         u=constant is a solution to A*u=0.  In that case, we regularize the problem
         by requiring mean(u)==bonus_in, and in return the f applied to get that
         value is really f-bonus_out (on interior terms).  This seems a bit picky,
         but it's required to make gmres behave */
      /* This is the public facing implementation, which load-balances f and u
         and calls cycle_p */
      void cycle(CYCLE_TYPE cycle, blitz::Array<double,2> & f, blitz::Array<double,2> & u,
            double bonus_in, double & bonus_out,int pre = 2, int mid = 2, int post = 2);
      
//   protected:
      /* Local array size */
      int local_size_x;
      int local_x_lbound, local_x_ubound;
      int coarse_x_lbound, coarse_x_ubound;

      int size_x, size_z;

      // Parallelization parameters
      int myrank, nproc;

      // Parameters for the coarse-grid solving
      bool coarse_symbolic_ok, // Whether the coarse grid symbolic factorizing is okay
           coarse_numeric_ok, // Same with the numerical factor
           // And parameters to control when a symbolic factor is not okay
           any_dxz, // Whether there's any nonzero dxz terms in the domain
           bc_tangent, // Whether there's any tangent derivatives in the BCs
           bc_normal; // Same, normal derivatives

      // numeric and symbolic factors from UMFPACK
      void * numeric_factor, * symbolic_factor; 

      // Arrays for the sparse representation of the coarse operator, see UMFPACK
      // documentation for the full information.  The short version is that
      // the entry A_double[A_cols[i]+j] is in column i, and row A_rows[A_cols[i]+j]
      int sparse_size; // number of entries in sparse array
      int * A_cols, // Array of column indices
          * A_rows; // Array of row index entries
      double * A_double; // Array of entries in the sparse matrix

      // Builds the sparse operator, storing proper entries in the above arrays
      // and allocating if necessary
      void build_sparse_operator();

      // Performs consistency checks on whether a symbolic coarse grid factor
      // is OK based on current conditions and the coarse-grid boolean parameters
      // (see above)
      void check_bc_consistency();
      
      /* One dimensional FD operators */
      /* Interpretation:
         Op(i,j) is the operator for point i, referring to:
            j=0 -- left (minus) neighbour
            j=1 -- self
            j=2 -- right (plus) neighbour
      */
      blitz::Array<double,2> Dx, Dxx, Dz, Dzz;

      /* Local coefficients for the 2D operator */
      blitz::Array<double,2> uxx, uzz, uxz, ux, uz;

      /* Local arrays for load-balanced f and u arrays, since those
         may differ from input arrays */
      blitz::Array<double,2> f_balance, u_balance;
      
      /* c, for the c*U term in the equation */
      double helm_parameter;
      
      /* Local bc coefficients */
      blitz::Array<double,1> u_left, u_right, ux_left, ux_right, uz_left, uz_right,
                           u_top, u_bot, ux_top, ux_bot, uz_top, uz_bot;

      /* True if the problem is indefinite, such that A*(constant) = 0.  
         See comments for cycle() for the implications */
      bool indefinite_problem;

      /* Symmetry type */
      SYM_TYPE symmetry_type;
      
      /* Perform one red-black relaxation pass */
      void do_redblack(blitz::Array<double,2> & f, blitz::Array<double,2> & u);

      /* Apply the operator "forwards" */
      void apply_operator(blitz::Array<double,2> & u, blitz::Array<double,2> & f);
               
      /* Coarse u, f arrays for multilevel problem */
      blitz::Array<double,2> coarse_u, coarse_f;

      /* Local, nonbalanced coarsened arrays.  Since the array gets roughly halved,
         we can either rebalance and then coarsen or coarsen and then rebalance.
         This takes the latter approach */
      blitz::Array<double,2> local_coarse_2d;
      blitz::Array<double,1> local_coarse_1d;

      /* If we coarsen an even number of points (odd number of
         intervals), then the "include every other point" strategy
         doesn't work (given that we keep the boundaries).  We have
         to thus pick an interval to keep between coarsening levels;
         keeping the largest interval on the fine grid seems to be
         an excellent choice */
      int kept_interval;
      bool coarsest_level;

      /* (semi)-coarsen a variable on a 2D grid.  The extra 
         boolean (even) tells the coarsening operator to 
         treat the problem as having even symmetry (if any) 
         rather than whatever has been specified.  This is 
         useful for the coefficients and boundary conditions, 
         since they'll never have odd symmetry */
      /* This operator does not rebalance -- it will simply write
         into local_coarse_2d (1d for line version) */
      void coarsen_grid(blitz::Array<double,2> & q, bool even=false);
      /* And do the same to a line.  */
      void coarsen_line(blitz::Array<double,1> & q, bool even=false);

      /* Interpolate (from local_coarse_2d) to full-sized array */
      void interpolate_grid(blitz::Array<double,2> & q_fine);
      // Interpolate line commented out -- I don't see a need for it
//      void interpolate_line(blitz::Array<double,1> & q_coarse,
//            blitz::Array<double,1> & q_fine, bool even=false);
      /* Local cycle, which implements the public function but assumes all relevant
         arrays are properly load-balanced */
      void _cycle(CYCLE_TYPE cycle, blitz::Array<double,2> & f, blitz::Array<double,2> & u,
            double bonus_in, double & bonus_out,int pre, int mid, int post);

      MPI_Comm my_comm;
      MPI_Comm coarse_comm; // Communicator for coarse grid
      MG_Solver * coarse_solver; // Coarse solver
};

/* Build a 1D finite-difference operator for arbitrary x-grid, see mkfd.m
   as reference in MATLAB */
void get_fd_operator(blitz::Array<double,1> & x, blitz::Array<double,2> & Dx,
      blitz::Array<double,2> & Dxx);

/* Get the local splitting of N points in the multigrid context.  This might (will)
   differ from the splitting in spectral context, because we want a minimum of 2 lines
   per processor, and preferably each processor will have an even number of lines
   to minimize communication.  e.g.
   4 4 4 4 6 is preferable to
   4 4 4 5 5, since the latter will mix up red/black identities
   (RBRB-RBRB-RBRB-RBRB-RBRBRB vs.
    RBRB-RBRB-RBRB-RBRBR-BRBRB -- the latter will involve a double-stall on 
                                  processor 4 for boundary data */
void get_local_split(int extent, int rank, int nproc, int & l_lbound, int & l_ubound);

/* Rebalances an array; an array that is orignally spit in orig is
   reassigned to balance, such that the global (compressed) array 
   is exactly the same */
void rebalance_array(blitz::Array<double,2> & orig, 
      blitz::Array<double,2> & balance, MPI_Comm c);
void rebalance_line(blitz::Array<double,1> & o, blitz::Array<double,1> & b, MPI_Comm c);

/* Perform a line solve, which is the 1d problem:
   A(z)*u_zz + B(z)*u_z + C(z) = f(z), with
   a0*u+b0*u_z = f0 @ z=0 (bottom)
   a1*u+b1*u_z = f1 @ z=1 (top)

   Unlike the 1D gmres controller, we're not going to go to
   great lengths to preserve a factored operator.  Firstly,
   there's no chance of an indefinite 1D problem here, and
   secondly most of the time we're going to be doing a -lot-
   of line solves, and the only way to cache things would
   be to store all of those operators.  Bad for memory. */
void line_solve(blitz::Array<double,1> & u, blitz::Array<double,1> & f,
      blitz::Array<double,1> & A, blitz::Array<double,1> & B,
      blitz::Array<double,1> & C, double a0, double a1,
      double b0, double b1, blitz::Array<double,2> & Dz,
      blitz::Array<double,2> & Dzz);

#endif
