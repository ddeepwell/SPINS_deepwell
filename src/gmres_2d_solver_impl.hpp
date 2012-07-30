#include "TArray.hpp"
#include "T_util.hpp"
#include "gmres.hpp"
#include "Parformer.hpp"
#include "multigrid.hpp"
#include "grad.hpp"
#include <blitz/array.h>
#include <vector>
#include <set>
#include <mpi.h>

using TArrayn::DTArray;
using TArrayn::Grad;
using Transformer::Trans1D;
using Transformer::S_EXP;

using std::vector;
using std::set;

/* Private implementation header for the GMRES framework, for the 2D multigrid-based
   solver.  The class defined here will use the Grad class to apply the proper operator
   for:
      Lap (u) + H*u = f
   where the Laplacian is defined in a possibly mapped coordinate system.

   BCs supported are of the Dirichlet/Neumann/Robin type, with the restriction
   for now that the same basic BC operator is applied to all boundaries.  If
   the x dimension has symmetry (periodic/Sin/Cos), then the implied BCs are
   used instead for that.

   An indefinite problem is possible if H is 0 and the BCs do not have Dirichlet
   conditions.  This will happen with either Neumann-type BCs explicitly, or
   Neumann in the vertical (z) and periodic/cos symmetry in the horizontal (x).

   In such a case, this GMRES class solves the "bordered" problem, which extends
   the DE as:
      Lap (u) + H*u = f - sigma
      u_normal (bdy) = g - sigma
      mean(u) = 0

   This makes the resulting discretized system non-singular, and in so doing
   handles the "compatibility condition" whereby int_V(f) = int_C(g).  This
   comes up in particular when solving for pressure, since in 2D or in the
   mean-spanwise 3D mode there is no H term.  Sigma will almost certainly be
   small, however, since in the pressure case it largely comes from rounding error.
   (Physically, it would reflect compressibility/normal flow that pressure somehow
   cannot account for.)
*/


/* ubox and fbox reflect the same structure, both being <grid> plus <addt'l>,
   but I found in programming 1DGMRES that having to always consider "extra"
   as the paramater was less-than-intuitive. */

struct ubox {
   /* Struct to hold a u vector, along with a possible sigma incompatibility
      parameter. */
   DTArray * restrict gridbox;
   double sigma;
};

struct fbox {
   /* Struct to hold a f-vector, along with a mean condition.*/
   DTArray * restrict gridbox;
   double mean;
};

/* The solver class, inheriting from GMRES_Interface */
class Cheb_2dmg : public GMRES_Interface < fbox * , ubox *> {
   private:
      // Array sizes.  The actual arrays used here will be formally 3D, but with
      // a single spanwise variable -- (szx,1,szy) form -- split over the appropriate
      // set of processors
      int szx, szz;

      // Last solved x-transform type.  If/when this changes, the MG preconditioner
      // probably needs rebuilt.
      S_EXP type_x;

      // Temporary arrays.  These will be referenced as more appropriate names when
      // used
      DTArray * temp1, * temp2, * temp3;

      // C-ordered arrays for interfacing with the multigrid preconditioner
      Array<double,2> * r2d, * s2d;

      // 1D arrays for Neumann BCs
      Array<double,1> dx_top, dz_top, dx_bot, dz_bot, dx_left, dz_left, dx_right, dz_right;
      
      // Whether the currently-solved problem (and last-solved one) was indefinite
      bool indefinite_problem;

      // problem parameters -- BC and helmholtz
      double dir_zbc, neum_zbc, dir_xbc, neum_xbc, helm;
      
      // Gradient and preconditioner ops
      Grad * grad_3d; // Passed-in (3D) gradient op
      Grad * grad_2d; // 2D gradient op for calculations, Ny = 1
      MG_Solver * mg_precond;

      // Build the multigrid preconditioner
      void build_precond();
      // Reconfigure the MG preconditioner (bcs, symmetry)
//      void config_precond();
      
      // MPI Communicator
      MPI_Comm my_comm;

   public:
      // Constructor and destructor
      Cheb_2dmg(int size_x, int size_z, MPI_Comm c = MPI_COMM_WORLD);
      ~Cheb_2dmg();

      // Set gradient operator
      void set_grad(Grad * in_grad);
      // Set BCs for new solve, including x-expansion
      void set_bc(double h, double zdir, double zneum, S_EXP sym_type, double xdir, double xneum);

      // Operators for GMRES
      fbox * alloc_resid();
      ubox * alloc_basis();
      void free_resid(fbox * &);
      void free_basis(ubox * &);
      void matrix_multiply(ubox * & lhs, fbox * & rhs);
      void precondition(fbox * & rhs, ubox * & soln);
      double resid_dot(fbox * &, fbox * &);
      double basis_dot(ubox * &, ubox * &);
      void resid_copy(fbox * & copyto, fbox * & copyfrom);
      void basis_copy(ubox * & copyto, ubox * & copyfrom);
      void resid_fma(fbox * & lhs, fbox * & rhs, double scale);
      void basis_fma(ubox * & lhs, ubox * & rhs, double scale);
      void resid_scale(fbox * & vec, double scale);
      void basis_scale(ubox * & vec, double scale);

      bool noisy() const;

   private:
      // Allocation data
      vector<fbox *> r_free;
      vector<ubox *> b_free;
      set<fbox *> r_used;
      set<ubox *> b_used;
};
