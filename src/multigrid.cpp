#include "multigrid.hpp"
#include "Par_util.hpp"

#include <blitz/array.h>
#include <iostream>
#include <blitz/tinyvec-et.h>

#include "umfpack.h"

using namespace blitz;
using namespace std;

//#define COARSE_GRID_SIZE 512
#define COARSE_GRID_SIZE 64
#define SYNC(__x__) { for (int _i = 0; _i < nproc; _i++) { \
                     if (myrank == _i) { \
                        __x__; \
                     } MPI_Barrier(my_comm); } MPI_Barrier(my_comm);}

/* Prototype the DGESV LAPACK routine */
extern "C" {
   void dgesv_(int * n, int * nrhs, double * a, int * lda, 
         int * ipiv, double * b, int * ldb, int * info);
   void dgbsv_(int * n, int * kl, int * ku, int * nrhs,
         double * ab, int * ldab, int * ipiv, double * b,
         int * ldb, int * info);
}

void get_fd_operator(Array<double,1> & x, Array<double,2> & Dx, Array<double,2> & Dxx) {
   /* Builds FD derivative operators (to second order, save at boundaries)
      for the first and second derivatives, given an arbitrary grid x */
   /* For output, Dx(i,j) will contain the operator local to x(i).
      So Dx*f @ x=j == Dx(i,0)*f(j-1) + Dx(i,1)*f(j) + Dx(i,2)*f(j+1) */

   assert(x.lbound(firstDim) == 0 && x.extent(firstDim) >= 2);
   
   /* Assert that x and operator extents are the same */
   assert(x.lbound(firstDim) == Dx.lbound(firstDim) &&
         x.ubound(firstDim) == Dx.ubound(firstDim));
   assert(x.lbound(firstDim) == Dxx.lbound(firstDim) &&
         x.ubound(firstDim) == Dxx.ubound(firstDim));

   /* Assert that the operator bounds are proper */
   assert(Dx.lbound(secondDim) == 0 && Dx.ubound(secondDim) == 2 &&
         Dxx.lbound(secondDim) == 0 && Dxx.ubound(secondDim) == 2);

   /* The first point (boundary) is easy, since it's one-sided differencing,
      and Dxx does not exist */
   int lbound = 0; int ubound = x.ubound(firstDim);

   Dx(lbound,0) = 0; // No reference to the left of the leftmost boundary
   Dx(lbound,1) = -1/(x(lbound+1)-x(lbound));
   Dx(lbound,2) = 1/(x(lbound+1)-x(lbound));
   Dx(ubound,2) = 0; // And none to the right of the rightmost boundary
   Dx(ubound,1) = 1/(x(ubound)-x(ubound-1));
   Dx(ubound,0) = -1/(x(ubound)-x(ubound-1));
   // No Dxx at the boundaries
   Dxx(lbound,0) = Dxx(lbound,1) = Dxx(lbound,2) = 0;
   Dxx(ubound,0) = Dxx(ubound,1) = Dxx(ubound,2) = 0;
   
   /* Now, loop over interior points */
   /* Matrix operator to solve for FD coefficients */
   Array<double,2> M(3,3,blitz::columnMajorArray);
   /* "vector" for Dx, Dxx coefficients.  Input is [0 1 0;0 0 1] and output is
      the coefficients for Dx/Dxx */
   Array<double,2> vec(3,2,blitz::columnMajorArray);
   int IPIV[3]; // Pivot array for DGESV
   /* Other constants for DGESV */
   int N=3, NRHS=2, LDA=3, LDB=3, INFO=-1;
   for (int j = lbound+1; j<= ubound-1; j++) {
      /* First, build M using a Taylor series approach.  We'll define
         the constant, dx, and dxx terms via differences on x, and then
         solve for the weights that give [0 1 0] for Dx and [0 0 1] for Dxx */
      /* Left and right spacings */
      double xl = x(j-1)-x(j);
      double xr = x(j+1)-x(j);
      /* Now, build the Taylor series about x=x(j), and evaluate
         at j-1, j, and j+1 */
      /* f(x) = f(x_j) + f_x(x_j)*(x-x_j) + f_xx(x_j)*(x-x_j)^2/2 */
      /* First row of matrix -- constant terms */
      M(0,0) = M(0,1) = M(0,2) = 1;
      /* Second row -- Dx term */
      M(1,0) = xl; M(1,1) = 0; M(1,2) = xr;
      /* Third row, Dxx term */
      M(2,0) = xl*xl/2, M(2,1) = 0; M(2,2) = xr*xr/2;

      /* Now, the rhs vector */
      vec(0,0) = vec(0,1) = 0;
      vec(1,0) = 1; vec(1,1) = 0;
      vec(2,0) = 0; vec(2,1) = 1;

      /* Now, M\vec, AKA an ugly DGESV call */
//      std::cerr << M;
      dgesv_(&N, &NRHS, M.data(), &LDA, IPIV, vec.data(), &LDB, &INFO);
      assert(INFO == 0);

//      std::cerr << vec;

      /* And assign results to Dx/Dxx */
      Dx(j,0) = vec(0,0); Dxx(j,0) = vec(0,1);
      Dx(j,1) = vec(1,0); Dxx(j,1) = vec(1,1);
      Dx(j,2) = vec(2,0); Dxx(j,2) = vec(2,1);
   }
}

void get_local_split(int extent, int rank, int nproc, int & l_lbound, int & l_ubound) {
   /* Allocate evenly-split pairs to as many processors as possible.
      An odd number of points means that the last processor gets an extra */
   int npairs = extent/2;
   bool odd = extent % 2;
   // Each processor will get at least base_number pairs
   int base_number = npairs / nproc;
   int extra_pairs = npairs % nproc;
   // If extent < 2*nproc, not every processor will get points
   int last_processor = -1;
   // If there are more than enough pairs to go around, each processor gets some
   if (npairs >= nproc) {
      last_processor = nproc - 1;
   // Otherwise, only extra pairs are distributed
   } else {
      last_processor = extra_pairs - 1;
   }
   // Processors 0-(extra_pairs-1) will have a bonus pairing
   l_lbound = 2*(base_number*rank + (extra_pairs > rank ? rank : extra_pairs));
   l_ubound = l_lbound + 2*(base_number + (extra_pairs > rank))-1;
   if (odd && rank == last_processor)
      l_ubound += 1;
}

/* Rebalance an array such that the global structure is preserved, but
   orig and balance might have different splittings.  The assumptions here
   are fairly general, but we will require that orig/balance both have
   increasing lbound/ubounds over the processors and are non-overlapping */

/* As a performance note, this can be split into two functions -- the first
   constructs the mapping of communication (who talks to whom), and the
   second actually performs the communication from this mapping.  For any
   given case, we're probably going to re-use the balancing several times */
#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)<(y) ? (y) : (x))
      
void rebalance_line(Array<double,1> & orig, Array<double,1> & balance, MPI_Comm c) {
   TinyVector<int,2> o_base, o_extent;
   o_base(0) = orig.lbound(firstDim); o_extent(0) = orig.extent(firstDim);
   o_base(1) = 0; o_extent(1) = 1;
//   Array<double,2> o_2d(Range(orig.lbound(firstDim),orig.ubound(firstDim)),Range(0,0));   
   Array<double,2> o_2d(o_base,o_extent);
   TinyVector<int,2> b_base, b_extent;
   b_base(0) = balance.lbound(firstDim); b_extent(0) = balance.extent(firstDim);
   b_base(1) = 0; b_extent(1) = 1;
//   Array<double,2> b_2d(Range(balance.lbound(firstDim),balance.ubound(firstDim)),Range(0,0));
   Array<double,2> b_2d(b_base,b_extent);
   o_2d(Range::all(),0) = orig;
   rebalance_array(o_2d,b_2d,c);
   if (balance.extent(firstDim) > 0) balance = b_2d(Range::all(),0);
}
void rebalance_array(Array<double,2> & orig, Array<double,2> & balance, MPI_Comm c) {
   int o_lbound = orig.lbound(firstDim);
   int o_ubound = orig.ubound(firstDim);
   int b_lbound = balance.lbound(firstDim);
   int b_ubound = balance.ubound(firstDim);
   int size_z = orig.extent(secondDim);
   int myrank, nproc;
   double * orig_data; // Pointer to the original data
   static Array<double,2> * temp_array = 0;
   // Now, we need to ensure that orig and balance have the same storage ordering
   if (!all(orig.ordering() == balance.ordering())) {
      GeneralArrayStorage<2> bal_storage(balance.ordering(),true);
      if (temp_array && (!all(temp_array->lbound() == o_lbound) ||
               !all(temp_array->ubound() == o_ubound) ||
               !all(temp_array->ordering() == balance.ordering()))) {
         delete temp_array;
         temp_array = 0;
      }
      if (!temp_array)
         temp_array = new Array<double,2> (orig.lbound(),orig.extent(),bal_storage);
      *temp_array = orig;
      orig_data = temp_array->data();
   } else {
      orig_data = orig.data();
   }
      
   MPI_Comm_size(c,&nproc);
   MPI_Comm_rank(c,&myrank);
//   fprintf(stdout,"Rebalancing array, %d/%d\n",myrank,nproc);
   /* Construct a view of the array bounds shared by all processors */
   int o_lbounds[nproc], o_ubounds[nproc], b_lbounds[nproc], b_ubounds[nproc];
   MPI_Allgather(&o_lbound,1,MPI_INT,o_lbounds,1,MPI_INT,c);
   MPI_Allgather(&o_ubound,1,MPI_INT,o_ubounds,1,MPI_INT,c);
   MPI_Allgather(&b_lbound,1,MPI_INT,b_lbounds,1,MPI_INT,c);
   MPI_Allgather(&b_ubound,1,MPI_INT,b_ubounds,1,MPI_INT,c);
//   fprintf(stdout,"Rebalancng array with global dimensions %d-%d\n",o_lbounds[0],o_ubounds[nproc-1]);

   /* Now, construct the mapping of counts and displacements by processor */
   int s_counts[nproc], s_displs[nproc], r_counts[nproc], r_displs[nproc];
   for (int k = 0; k < nproc; k++) {
      /* If our orig array overlaps their balance array... */
      if (o_lbound <= b_ubounds[k] && o_ubound >= b_lbounds[k]) {
         /* We want to send the region from the lowest to highest points
            of overlap */
         /* The lowest possible index we can send is our lower bound, and
            the highest possible we want to start sending is the recipient's
            lower bound */
         int send_lb = MAX(b_lbounds[k],o_lbound);
         /* Reversing the logic of above, the highest possible index to
            send is our upper bound, and the highest we want to send is
            the recipient's upper bound */
         int send_ub = MIN(o_ubound,b_ubounds[k]);
         s_counts[k] = size_z*(1+send_ub-send_lb);
         s_displs[k] = (send_lb-o_lbound)*size_z;
      } else {
         s_counts[k] = s_displs[k] = 0;
      }
      /* And for receive, if our destination array overlaps their orig array... */
      if (b_lbound <= o_ubounds[k] && b_ubound >= o_lbounds[k]) {
         int rec_lb = MAX(b_lbound,o_lbounds[k]);
         int rec_ub = MIN(b_ubound,o_ubounds[k]);
         r_counts[k] = size_z*(1+rec_ub-rec_lb);
         r_displs[k] = (rec_lb-b_lbound)*size_z;
      } else {
         r_counts[k] = r_displs[k] = 0;
      }
   }

   // Finally, call MPI_Alltoallv to perform the balancing
#if 0 // debug output
   for (int qq = 0; qq < nproc; qq++) {
      if (qq == myrank) {
         fprintf(stderr,"%d: ",myrank);
         for (int k = 0; k < nproc; k++) {
            fprintf(stderr,"%d/%d ",s_counts[k],r_counts[k]);
         }
         fprintf(stderr,"\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
#endif
   
   MPI_Alltoallv(orig_data,s_counts,s_displs,MPI_DOUBLE,
         balance.data(),r_counts,r_displs,MPI_DOUBLE,c);
   // And done
}
void MG_Solver::set_x_symmetry(SYM_TYPE s) {
   // If previous symmetry type was different, then the numeric factor is bad
   if (symmetry_type != s) coarse_numeric_ok = false;
   symmetry_type = s;
   // Propagate change to the coarse operator
   if (coarse_solver) coarse_solver->set_x_symmetry(s);

   // And recheck BC consistency, since an odd->even change can make this
   // an indefinite problem
   if (coarsest_level)
      check_bc_consistency();
}
/* Construct local arrays, resizing the blitz::Arrays to their proper size */
MG_Solver::MG_Solver(Array<double,1> xvals, blitz::Array<double,1> zvals,
      SYM_TYPE sym, MPI_Comm c):
   coarse_solver(0), coarse_x_lbound(0), coarse_x_ubound(-1),
   coarse_symbolic_ok(false), coarse_numeric_ok(false), any_dxz(false),
   bc_tangent(false), bc_normal(false), numeric_factor(0), symbolic_factor(0),
   sparse_size(0), A_rows(), A_cols(), A_double(),
   coarsest_level(false)
{
   my_comm = c;
   symmetry_type = sym;
   MPI_Comm_size(c,&nproc); MPI_Comm_rank(c,&myrank);
   /* Get the local sizings, and find out what section of the array we're
      actually responsible for */
   size_x = xvals.extent(firstDim) - (sym?2:0);
   size_z = zvals.extent(firstDim);
   get_local_split(size_x,myrank,nproc,local_x_lbound,local_x_ubound);
   local_size_x = local_x_ubound - local_x_lbound +1;

   /* Resize and re-index the arrays, so that they point to the local section
      of the global operator space */
   TinyVector<int,2> base_vector; 
   base_vector(0) = local_x_lbound;
   base_vector(1) = 0;
   uxx.resize(local_size_x,size_z); uxx.reindexSelf(base_vector);
   uzz.resize(local_size_x,size_z); uzz.reindexSelf(base_vector);
   uxz.resize(local_size_x,size_z); uxz.reindexSelf(base_vector);
   ux.resize(local_size_x,size_z); ux.reindexSelf(base_vector);
   uz.resize(local_size_x,size_z); uz.reindexSelf(base_vector);

   /* Resize Dx/Dxx to include only the local part of the tensor product */
   Dx.resize(local_size_x,3); Dx.reindexSelf(base_vector);
   Dxx.resize(local_size_x,3); Dxx.reindexSelf(base_vector);

   /* Top and bottom BCs need resized in the same manner */
   TinyVector<int,1> base_1d; base_1d(0) = local_x_lbound;
   u_bot.resize(local_size_x); u_bot.reindexSelf(base_1d);
   ux_bot.resize(local_size_x); ux_bot.reindexSelf(base_1d);
   uz_bot.resize(local_size_x); uz_bot.reindexSelf(base_1d);
   u_top.resize(local_size_x); u_top.reindexSelf(base_1d);
   ux_top.resize(local_size_x); ux_top.reindexSelf(base_1d);
   uz_top.resize(local_size_x); uz_top.reindexSelf(base_1d);

   u_left.resize(size_z); u_right.resize(size_z);
   ux_left.resize(size_z); ux_right.resize(size_z);
   uz_left.resize(size_z); uz_right.resize(size_z);

   /* Dz/Dzz don't have any local/global pairing */
   Dz.resize(size_z,3); Dzz.resize(size_z,3);

   /* Compute global Dx/Dxx operators */
   Array<double,2> global_Dx(Range(xvals.lbound(firstDim),xvals.ubound(firstDim)),Range(0,2));
   Array<double,2> global_Dxx(Range(xvals.lbound(firstDim),xvals.ubound(firstDim)),Range(0,2));
   get_fd_operator(xvals,global_Dx,global_Dxx);

   /* Assign a slice of the global FD operator to our local view.  If there is symmetry,
      the relevant indicies shift by one because x is assumed to include ghost points
      reflecting the boundaries */
   Dx = global_Dx(Range(local_x_lbound +(sym?1:0) ,local_x_ubound + (sym?1:0)),Range(0,2));
   Dxx = global_Dxx(Range(local_x_lbound + (sym?1:0),local_x_ubound + (sym?1:0)),Range(0,2));

   /* There aren't any such complexities for z */
   get_fd_operator(zvals,Dz,Dzz);

   // To coarsen, we'll about halve the number of interior points, keeping boundaries.
   // Because of the ghost point deal, we will always have "boundaries" in the x_line.
   // So, the number of interior points is equal to x_line.extent() - 2
   int interior_points = xvals.extent(firstDim)-2;
   
   // If there are only three x-points, we're at the coarsest level.  This represents
   // one interior point and two boundaries.
//   if (interior_points <= 3) return;
   if ((nproc == 1) && sym != SYM_NONE && interior_points <= COARSE_GRID_SIZE) {
      coarsest_level = true;
      if (myrank == 0) {
         A_cols = new int[size_x*size_z+2];
         A_rows = new int[size_x*size_z*11];
         A_double = new double[size_x*size_z*11];
//         A_rows = new int[10*((size_x-2)*(size_z-2))+5*(2*(size_x-2)+2*(size_z-2))+16+size_x*size_z];
//         A_double = new double[10*((size_x-2)*(size_z-2))+5*(2*(size_x-2)+2*(size_z-2))+16+size_x*size_z];
      }
      return;
   }
   else if ((nproc == 1) && interior_points <= (COARSE_GRID_SIZE-2)) {
      coarsest_level = true;
      if (myrank == 0) {
         A_cols = new int[size_x*size_z+2];
         A_rows = new int[11*(size_x*size_z)];
         A_double = new double[11*(size_x*size_z)];
      }
      return;
   }

   // Otherwise, we want to build a coarse-grid operator
   // The number of interior points is halved, rounding down 
   int size_coarse_x = 2+(interior_points)/2;
   if (sym != SYM_NONE) size_coarse_x = 2+(interior_points+1)/2;
   Array<double,1> coarse_x(size_coarse_x);

//   fprintf(stderr,"Coarse x has size %d\n",size_coarse_x);

   if (sym != SYM_NONE || interior_points % 2 == 1) {
      /* If there's an odd number of interior points, we're set -- just include
         every other one, and we're done modulo smmetry and ghost points */
      if (sym == SYM_NONE) {
         for (int i = 0; i < size_coarse_x; i++)
            coarse_x(i) = xvals(2*i);
      } else {
         for (int i = 0; i < size_coarse_x-2; i++)
            coarse_x(i+1) = xvals(2*i+1);
      }
      kept_interval = -1;
   } else {
      /* If there's an even number of interior points, we have a finer decision
         to make.  We can't evenly halve an odd number of intervals, so we
         get to pick the location where we break our "every other point"
         coarsening.  EG, go from
         x-x-x-x-x (where x is kept, n==9 here) to
         x-x-xx-x-x (n == 10, n==8 interior)
      */
      double dx = 0;
      for (int i = 0; i < size_x-1 + (sym?2:0); i+=2) {
         if (fabs(xvals(i+1)-xvals(i)) > dx) {
            dx = fabs(xvals(i+1)-xvals(i));
            kept_interval = i;
         }
      }
//      fprintf(stderr,"Kept interval is %d\n",kept_interval);
      /* Now, with the kept interval in mind, keep every 2nd point until the
         magic interval, then the point right after said interval, then
         every 2nd point thereafter */
      if (sym == SYM_NONE) {
         for (int i = 0; i < size_coarse_x; i++) 
            coarse_x(i) = xvals(2*i - (2*i > kept_interval));
      } else {
//         for (int i = 0; i < size_coarse_x-2; i++) {
//            coarse_x(i+1) = xvals(2*i + 1 - ((2*i) > kept_interval));
//         }
//         kept_interval -= 1;
      }
         
   }
   /* If we have symmetry, we need to correct the left and right points
      (the "ghost boundary points") to reflect our new interiors.  If the problem
      has symmetry, then the first and last points in xvals reflect the location
      of the second points, mirrored about the true boundary.  We want to keep
      this new mirroring */
   if (sym!= SYM_NONE) {
      if (sym != SYM_PERIODIC) {
         double left_bdy = 0.5*(xvals(0)+xvals(1));
         double right_bdy = 0.5*(xvals(size_x)+xvals(size_x+1));
         coarse_x(0) = coarse_x(1)-2*(coarse_x(1)-left_bdy);
         coarse_x(size_coarse_x-1) = 2*right_bdy - coarse_x(size_coarse_x-2);
         // Also, tweak the "kept interval", since we defined it in the
         // x-including-ghost-point frame and that is no longer appropriate
//         kept_interval -= 1;
      } else { // Periodic
         // In the periodic case, it's kosher to shift the domain boundaries
         // by a fixed amount.  This ensures accuracy in derivatives, since
         // the right ghost point can reflect the true location of the left-most
         // point and vice versa.  Otherwise, differentiation would be inconsistent
         double left_bdy = 0.5*(xvals(0)+xvals(1));
         double right_bdy = 0.5*(xvals(size_x)+xvals(size_x+1));
         double mind_gap = (coarse_x(1)-left_bdy) + (right_bdy - coarse_x(size_coarse_x-2));
         coarse_x(0) = coarse_x(1)-mind_gap;
         coarse_x(size_coarse_x-1) = coarse_x(size_coarse_x-2)+mind_gap;
//         kept_interval -= 1;
      }
   }
         
//   if (myrank == 0) {
//      fprintf(stdout,"x:\n");
//      cout << xvals;
//      fprintf(stdout,"Coarse x:\n");
//      cout << coarse_x;
//   }
//   MPI_Barrier(my_comm);
   // Now, find out what distribution of points we'll have on the coarse array
   int c_extent = coarse_x.extent(firstDim) - (sym ? 2 : 0);
   get_local_split(c_extent,myrank,nproc,coarse_x_lbound,coarse_x_ubound);
   // If the number of coarse points is less than the coarse-grid-size,
   // dump it all on the first processor so we can go ahead with the
   // sparse solve
   if (c_extent <= COARSE_GRID_SIZE) {
      if (myrank == 0) {
         coarse_x_lbound = 0;
         coarse_x_ubound = c_extent - 1;
      } else {
         coarse_x_lbound = c_extent;
         coarse_x_ubound = c_extent - 1;
      }
   }

   // Rebase the coarse_x and coarse_f arrays
   coarse_u.resize(coarse_x_ubound-coarse_x_lbound+1,size_z);
   coarse_u.reindexSelf(TinyVector<int,2>(coarse_x_lbound,0));
   coarse_f.resize(coarse_x_ubound-coarse_x_lbound+1,size_z);
   coarse_f.reindexSelf(TinyVector<int,2>(coarse_x_lbound,0));

   // And resize/rebase the local coarse arrays

   // Need to know whether we have the kept point, if any
   bool have_kept_interval = (local_x_lbound <= kept_interval) && (local_x_ubound >= kept_interval);
   
   // Define the local coarse extent
   int local_coarse_extent = local_size_x/2 + // Half the local size, plus...
                           have_kept_interval + // if we have the kept point, plus..
                           // whether we're the last proc with an odd number of points
                           ((myrank == nproc-1) && (local_size_x%2)); 
                            // (both of these won't be true at once)
   int local_coarse_lbound = local_x_lbound/2 + (kept_interval >= 0 && local_x_lbound > kept_interval);
//   fprintf(stdout,"COARSE: %d %d %d\n",myrank,local_coarse_lbound,local_coarse_extent);
   local_coarse_2d.resize(local_coarse_extent,size_z);
   local_coarse_1d.resize(local_coarse_extent);
   local_coarse_2d.reindexSelf(TinyVector<int,2>(local_coarse_lbound,0));
   local_coarse_1d.reindexSelf(TinyVector<int,1>(local_coarse_lbound));

//   for (int i = 0; i < nproc; i++) {
//      if (myrank == i)
//         fprintf(stdout,"Process %d has %d to %d on coarse array\n",myrank,coarse_x_lbound,coarse_x_ubound);
//      MPI_Barrier(my_comm);
//   }

   // Now, it's possible that not all processors will be used on the coarse
   // level -- that's natural as the number of x-lines approaches 3 (one
   // interior line).  So we'll want to -split- the communicator and only
   // make the coarser-level-object if this node has something to do there.

   if (coarse_x_ubound >= coarse_x_lbound) {
      MPI_Comm_split(my_comm,1,0,&coarse_comm);
      int sz2; MPI_Comm_size(coarse_comm,&sz2);
//      for (int i = 0; i < sz2; i++) {
//         if (i == myrank)
//            fprintf(stdout,"Process %d is recursing\n",myrank);
//         MPI_Barrier(coarse_comm);
//      }
      coarse_solver = new MG_Solver(coarse_x,zvals,sym,coarse_comm);
   } else {
      MPI_Comm_split(my_comm,0,0,&coarse_comm);
      coarse_solver = 0;
   }
   
}
  
/* Line solve */
void line_solve(Array<double,1> & u, Array<double,1> & f,
      Array<double,1> & A, Array<double,1> & B,
      Array<double,1> & C, double a0, double a1, 
      double b0, double b1, Array<double,2> & Dz,
      Array<double,2> & Dzz) {
   /* The objective here is to build the operator as a (tridiagonal) banded
      matrix for LAPACK, and to solve that using dgbsv (general band solver)
      to get u from the specified f.  We will be rebuilding the operator
      at each call (but not reallocating the band matrix) because we're
      likely to see this called multiple times with different parameters
      (especially C).  For reference, the problem is:
         A(z)*Dzz*u+B(z)*Dz*u + C(z)*u = f(z), with
         a0*u+b0*Dz*u = f0 @ z=0 (bottom), and
         a1*u+b1*Dz*u = f1 @ z=1 (top)
   */
   /* Allocate the operator as static, since it's large-ish */
//   cout << "Solving: A: " << A << "B: " << B << "C: " << C << "f: " << f;
//   fprintf(stdout,"BCs: %f %f, %f %f\n",a0,a0,b0,b1);
   blitz::firstIndex ii; blitz::secondIndex jj;
   static Array<double,2> band_matrix(4,u.extent(firstDim),blitz::columnMajorArray);
   band_matrix = 0;
   int IPIV[u.extent(firstDim)]; // Integer pivots for solve
   if (band_matrix.extent(secondDim) != u.extent(firstDim)) {
      // If our size has somehow changed, resize the band matrix to suit
      band_matrix.resize(4,u.extent(firstDim)); 
   }
   /* Set the band matrix based on our input parameters */
   // Main diagonal
   int top = band_matrix.ubound(secondDim);
   band_matrix(2,Range::all()) = A(ii)*Dzz(Range::all(),1) + 
      B(ii)*Dz(Range::all(),1) + C(ii);
//   // Subdiagonal
   band_matrix(3,Range(0,top-1)) = A(Range(1,top))*Dzz(Range(1,top),0) +
      B(ii)*Dz(Range(1,top),0);
//   // Superdiagonal
   band_matrix(1,Range(1,top)) = A(Range(0,top-1))*Dzz(Range(0,top-1),2) +
      B(ii)*Dz(Range(0,top-1),2);

   /* BC's */
//   band_matrix(Range::all(),0) = 0;
//   band_matrix(Range::all(),top) = 0;

   /* Bottom -- a0*u+b0*Dz*u */
   band_matrix(2,0) = a0 + b0*Dz(0,1); // Main diagonal from Dz
   band_matrix(1,1) = b0*Dz(0,2); // superdiagonal from Dz
   band_matrix(2,top) = a1 + b1*Dz(top,1);
   band_matrix(3,top-1) = b1*Dz(top,0); // Subdiagonal
   
   /* Parameters for LAPACK call */
   int N=f.extent(firstDim), KL=1, KU=1, NRHS=1, LDAB=4, LDB=N, INFO=-1;
   // dgbsv overwrites f with u, so copy
   u = f;
   // LAPACK call
//   cout << "Band matrix: " << band_matrix;
   dgbsv_(&N, &KL, &KU, &NRHS, band_matrix.data(), &LDAB, IPIV, u.data(), &LDB, &INFO);
//   cout << "Returning u:" << u << "Info " << INFO << endl;
   if (INFO != 0) {
      cerr << "Bad line solve, no cookie! (" << INFO << ")\n";
      cerr << A << B << C;
      cerr << band_matrix;
      cerr << f;
   }
   assert(INFO == 0);
}

/* Copy coefficients to the local object, and coarsen for the subproblem */
void MG_Solver::problem_setup(Array<double,2> & Uxx, Array<double,2> & Uzz,
      Array<double,2> & Uxz, Array<double,2> & Ux, Array<double,2> & Uz) {
   coarse_numeric_ok = false;
   rebalance_array(Uxx,uxx,my_comm);
   rebalance_array(Uzz,uzz,my_comm);
   rebalance_array(Ux,ux,my_comm);
   rebalance_array(Uz,uz,my_comm);
//   SYNC(cerr << Uxz; cerr.flush());
   rebalance_array(Uxz,uxz,my_comm);
//   SYNC(cerr << uxz; cerr.flush());
//   MPI_Finalize(); exit(1);
   /* Do coarsening */
   Array<double,2> CUxx, CUzz, CUxz, CUx, CUz;
//   int csize = coarse_x_ubound-coarse_x_lbound+1;
   int csize = coarse_u.extent(firstDim);
//   fprintf(stdout,"Coarse size %d\n",csize);
//   MPI_Barrier(my_comm);
//   if (csize <= 0) return;
   if (coarsest_level) {
      coarse_numeric_ok = false;
      if (any(Uxz!=0)) {
         if (!any_dxz) coarse_symbolic_ok = false;
         any_dxz = true;
      } else {
         if (any_dxz) coarse_symbolic_ok = false;
         any_dxz = false;
      }
      return;
   }
   TinyVector<int,2> cbase(coarse_x_lbound,0);
   CUxx.resize(csize,size_z); CUxx.reindexSelf(coarse_u.lbound());
   CUzz.resize(csize,size_z); CUzz.reindexSelf(coarse_u.lbound());
   CUxz.resize(csize,size_z); CUxz.reindexSelf(coarse_u.lbound());
   CUx.resize(csize,size_z); CUx.reindexSelf(coarse_u.lbound());
   CUz.resize(csize,size_z); CUz.reindexSelf(coarse_u.lbound());
//   fprintf(stdout,"Coarsening Uxx, size %dx%d\n",Uxx.extent(firstDim),Uxx.extent(secondDim));
   coarsen_grid(uxx,true); 
//   fprintf(stdout,"Rebalancing coarsened Uxx\n");
   rebalance_array(local_coarse_2d,CUxx,my_comm);
   coarsen_grid(uxz,true); rebalance_array(local_coarse_2d,CUxz,my_comm);
   coarsen_grid(uzz,true); rebalance_array(local_coarse_2d,CUzz,my_comm);
   coarsen_grid(ux,true); rebalance_array(local_coarse_2d,CUx,my_comm);
   coarsen_grid(uz,true); rebalance_array(local_coarse_2d,CUz,my_comm);
   if (coarse_solver)
      coarse_solver->problem_setup(CUxx,CUzz,CUxz,CUx,CUz);
}

void MG_Solver::helmholtz_setup(double h) {
   coarse_numeric_ok = false;
   helm_parameter = h;
   if (coarsest_level)
      check_bc_consistency();
   if (coarse_solver)
      coarse_solver -> helmholtz_setup(h);
}

void MG_Solver::bc_setup(int dim, Array<double,1> u_min, Array<double,1> uz_min,
      Array<double,1> ux_min, Array<double,1> u_max, Array<double,1> uz_max,
      Array<double,1> ux_max) {
   /* Copy BC data to the local object, and punt down the chain for a
      cosarse problem */
   coarse_numeric_ok = false;
//   MPI_Finalize(); exit(1);
   if (dim == 0) {
      // x -- no balancing needed
      u_left = u_min; u_right = u_max;
      ux_left = ux_min; ux_right = ux_max;
      uz_left = uz_min; uz_right = uz_max;
      if (coarsest_level) {
         check_bc_consistency();
      }
      if (coarse_solver)
         coarse_solver->bc_setup(dim,u_min,uz_min,ux_min,u_max,uz_max,ux_max);
   } else if (dim == 1) {
      // z -- balance
      Array<double,2> bc_incoming(Range(u_min.lbound(firstDim),u_min.ubound(firstDim)),Range(0,5));
      Array<double,2> bc_here(Range(local_x_lbound,local_x_ubound),Range(0,5));
      bc_here = 0;
      bc_incoming(Range::all(),0) = u_min;
      bc_incoming(Range::all(),1) = u_max;
      bc_incoming(Range::all(),2) = uz_min;
      bc_incoming(Range::all(),3) = uz_max;
      bc_incoming(Range::all(),4) = ux_min;
      bc_incoming(Range::all(),5) = ux_max;
      rebalance_array(bc_incoming,bc_here,my_comm);
      u_bot = bc_here(Range::all(),0);
      u_top = bc_here(Range::all(),1);
      uz_bot = bc_here(Range::all(),2);
      uz_top = bc_here(Range::all(),3);
      ux_bot = bc_here(Range::all(),4);
      ux_top = bc_here(Range::all(),5);
      // Coarsen

      Array<double,1> cu_min, cuz_min, cux_min, cu_max, cux_max, cuz_max;
      int c_size = coarse_x_ubound-coarse_x_lbound+1;
//      if (c_size <= 0) return; // coarsest grid
      if (coarsest_level) {
//         fprintf(stdout,"Coarsest level, so not recursing\n");
         check_bc_consistency();
         return;
      }
      TinyVector<int,1> c_base(coarse_x_lbound);
//      fprintf(stdout,"Line BC, size %d base %d\n",c_size,c_base(0));
      cu_min.resize(c_size); cu_min.reindexSelf(c_base);
      cuz_min.resize(c_size); cuz_min.reindexSelf(c_base);
      cux_min.resize(c_size); cux_min.reindexSelf(c_base);
      cu_max.resize(c_size); cu_max.reindexSelf(c_base);
      cux_max.resize(c_size); cux_max.reindexSelf(c_base);
      cuz_max.resize(c_size); cuz_max.reindexSelf(c_base);
      coarsen_line(u_bot); rebalance_line(local_coarse_1d,cu_min,my_comm);
      coarsen_line(u_top); rebalance_line(local_coarse_1d,cu_max,my_comm);
      coarsen_line(ux_bot); rebalance_line(local_coarse_1d,cux_min,my_comm);
      coarsen_line(ux_top); rebalance_line(local_coarse_1d,cux_max,my_comm);
      coarsen_line(uz_bot); rebalance_line(local_coarse_1d,cuz_min,my_comm);
      coarsen_line(uz_top); rebalance_line(local_coarse_1d,cuz_max,my_comm);
      if (coarse_solver) 
         coarse_solver->bc_setup(dim,cu_min,cuz_min,cux_min,cu_max,cuz_max,cux_max);
         
   }
}
   
   

void MG_Solver::apply_operator(Array<double,2> & u, Array<double,2> & f) {
   /* Assuming that the x-array is properly load balanced for our communicator,
      apply the operator A*x = f */
   /* Local arrays for left and right x lines, for parallel communication */
   Array<double,1> uin_left(size_z) ,  uin_right(size_z) ;
   uin_left = -9e9; uin_right = -9e9;

   /* Get rank and size of MPI Communicator */

   if (nproc > 1 || symmetry_type == SYM_PERIODIC) {
      /* We'll have to communicate with neighbours */
      int lefty = myrank-1;
      int righty = myrank+1;
      MPI_Status ignoreme;
      /* If we're the leftmost processor and we're not a periodic problem,
         then we don't need to communicate with anybody for our left point */
      if (myrank == 0 && symmetry_type != SYM_PERIODIC) {
         lefty = MPI_PROC_NULL;
      } else if (myrank == 0) {
         lefty = nproc - 1;
      }
      /* Same for rightmost processor and right point */
      if (myrank == nproc - 1 && symmetry_type != SYM_PERIODIC) {
         righty = MPI_PROC_NULL;
      } else if (myrank == nproc - 1){
         righty = 0;
      }
      /* Now, send left points, and receive right points */
//      fprintf(stderr,"%d sending left (%d, %d)\n",myrank,lefty,righty);
      MPI_Sendrecv(&u(u.lbound(firstDim),0),size_z,MPI_DOUBLE,lefty,0,
            uin_right.data(),size_z,MPI_DOUBLE,righty,0,my_comm,&ignoreme);
      /* And vice versa -- send right, receive left */
//      fprintf(stderr,"%d sending right (%d, %d)\n",myrank,righty,lefty);
      MPI_Sendrecv(&u(u.ubound(firstDim),0),size_z,MPI_DOUBLE,righty,0,
            uin_left.data(),size_z,MPI_DOUBLE,lefty,0,my_comm,&ignoreme);
//      fprintf(stderr,"%d received:\n");
//      fprintf(stderr,"%d done\n",myrank);
   }
   /* Fake left/right boundaries for even/odd symmetry */
   if (myrank == 0 && symmetry_type == SYM_EVEN) {
      uin_left = u(local_x_lbound,Range::all());
   } else if (myrank == 0 && symmetry_type == SYM_ODD) {
      uin_left = -u(local_x_lbound,Range::all());
   }
   /* Right */
   if (myrank == nproc - 1 && symmetry_type == SYM_EVEN) {
      uin_right = u(local_x_ubound,Range::all());
   } else if (myrank == nproc - 1 && symmetry_type == SYM_ODD) {
      uin_right = -u(local_x_ubound,Range::all());
   }
   /* By this point, we either have boundary data in uin_left/uin_right or
      we don't need it on account of being the boundary */
   /* We'll deal with the boundary points separately, since they need
      special tender love and care.  However, the interior is relatively
      easy */

   blitz::firstIndex ii; blitz::secondIndex jj;
   for (int i = local_x_lbound+1; i <= local_x_ubound-1; i++) {
      for (int j = 1; j <= size_z-2; j++) {
         f(i,j) = 
            u(i,j)*(helm_parameter + 
                  ux(i,j)*Dx(i,1) + uz(i,j)*Dz(j,1) +
                  uxx(i,j)*Dxx(i,1) + uzz(i,j)*Dzz(j,1) +
                  uxz(i,j)*Dx(i,1)*Dz(j,1)) +
            u(i-1,j)*(
                  ux(i,j)*Dx(i,0) + uxx(i,j)*Dxx(i,0) +
                  uxz(i,j)*Dx(i,0)*Dz(j,1)) +
            u(i+1,j)*(
                  ux(i,j)*Dx(i,2) + uxx(i,j)*Dxx(i,2) +
                  uxz(i,j)*Dx(i,2)*Dz(j,1)) +
            u(i,j-1)*(
                  uz(i,j)*Dz(j,0) + uzz(i,j)*Dzz(j,0) +
                  uxz(i,j)*Dx(i,1)*Dz(j,0)) +
            u(i,j+1)*(
                  uz(i,j)*Dz(j,2) + uzz(i,j)*Dzz(j,2) +
                  uxz(i,j)*Dx(i,1)*Dz(j,2)) +
            uxz(i,j)*(u(i-1,j-1)*Dx(i,0)*Dz(j,0) +
                  u(i+1,j-1)*Dx(i,2)*Dz(j,0) +
                  u(i-1,j+1)*Dx(i,0)*Dz(j,2) +
                  u(i+1,j+1)*Dx(i,2)*Dz(j,2));
      }
   }
   /* Top and bottom BCs */
   for (int i = local_x_lbound+1; i <= local_x_ubound - 1; i++) {
      int j = 0;
      f(i,j) = u(i,j)*(u_bot(i) + ux_bot(i)*Dx(i,1) + uz_bot(i)*Dz(j,1)) +
            u(i-1,j)*ux_bot(i)*Dx(i,0) + u(i+1,j)*ux_bot(i)*Dx(i,2) +
            u(i,j+1)*uz_bot(i)*Dz(j,2);
//      fprintf(stderr,"Bottom %d\n %g = %g*(%g + %g*%g + %g*%g) + %g*%g*%g + %g*%g*%g + %g*%g*%g\n",i,
//            f(i,j),u(i,j),u_bot(i),ux_bot(i),Dx(i,1),uz_bot(i),Dz(j,1),
//            u(i-1,j),ux_bot(i),Dx(i,0),u(i+1,j),ux_bot(i),Dx(i,2),
//            u(i,j+1),uz_bot(i),Dz(j,2));
//      fprintf(stderr,"Bottom %d %gL %gH %gR %gU = %g\n    (%g %g %g)\n",
//            i,u(i-1,j),u(i,j),u(i+1,j),u(i,j+1),f(i,j),
//            u_bot(i),ux_bot(i),uz_bot(i));;
//      fprintf(stderr,"(%d,%d) stencil:\n",i,j);
//      fprintf(stderr,"%g u, %g ux, %g uz\n",u_bot(i),ux_bot(i),uz_bot(i));
//      fprintf(stderr,"%g left %g here %g right %g up\n",u(i-1,j),u(i,j),u(i+1,j),u(i,j+1));
//      fprintf(stderr,"[%g %g %g]x [-- %g %g]z\n",Dx(i,0),Dx(i,1),Dx(i,2),Dz(j,1),Dz(j,2));
//      MPI_Finalize(); exit(1);
      j=size_z-1;
      f(i,j) = u(i,j)*(u_top(i) + ux_top(i)*Dx(i,1) + uz_top(i)*Dz(j,1)) +
            u(i-1,j)*ux_top(i)*Dx(i,0) + u(i+1,j)*ux_top(i)*Dx(i,2) +
            u(i,j-1)*uz_top(i)*Dz(j,0);
   }
   /* Now, handle non-boundary interiors */
   /* Left boundary */
   if (symmetry_type != SYM_NONE || (myrank != 0)) {
      int i = local_x_lbound;
      for (int j = 1; j <= size_z-2; j++) {
         f(i,j) = 
            u(i,j)*(helm_parameter + 
                  ux(i,j)*Dx(i,1) + uz(i,j)*Dz(j,1) +
                  uxx(i,j)*Dxx(i,1) + uzz(i,j)*Dzz(j,1) +
                  uxz(i,j)*Dx(i,1)*Dz(j,1)) +
            uin_left(j)*(
                  ux(i,j)*Dx(i,0) + uxx(i,j)*Dxx(i,0) +
                  uxz(i,j)*Dx(i,0)*Dz(j,1)) +
            u(i+1,j)*(
                  ux(i,j)*Dx(i,2) + uxx(i,j)*Dxx(i,2) +
                  uxz(i,j)*Dx(i,2)*Dz(j,1)) +
            u(i,j-1)*(
                  uz(i,j)*Dz(j,0) + uzz(i,j)*Dzz(j,0) +
                  uxz(i,j)*Dx(i,1)*Dz(j,0)) +
            u(i,j+1)*(
                  uz(i,j)*Dz(j,2) + uzz(i,j)*Dzz(j,2) +
                  uxz(i,j)*Dx(i,1)*Dz(j,2)) +
            uxz(i,j)*(uin_left(j-1)*Dx(i,0)*Dz(j,0) +
                  u(i+1,j-1)*Dx(i,2)*Dz(j,0) +
                  uin_left(j+1)*Dx(i,0)*Dz(j,2) +
                  u(i+1,j+1)*Dx(i,2)*Dz(j,2));
      }
      int j = 0;
      f(i,j) = u(i,j)*(u_bot(i) + ux_bot(i)*Dx(i,1) + uz_bot(i)*Dz(j,1)) +
            uin_left(j)*ux_bot(i)*Dx(i,0) + u(i+1,j)*ux_bot(i)*Dx(i,2) +
            u(i,j+1)*uz_bot(i)*Dz(j,2);
//      fprintf(stderr,"Bottom %d %gL %gH %gR %gU = %g\n",i,uin_left(j),u(i,j),u(i+1,j),u(i,j+1),f(i,j));
      j=size_z-1;
      f(i,j) = u(i,j)*(u_top(i) + ux_top(i)*Dx(i,1) + uz_top(i)*Dz(j,1)) +
            uin_left(j)*ux_top(i)*Dx(i,0) + u(i+1,j)*ux_top(i)*Dx(i,2) +
            u(i,j-1)*uz_top(i)*Dz(j,0);
   }
   /* Right boundary */
   if (symmetry_type != SYM_NONE || (myrank != nproc-1)) {
      int i = local_x_ubound;
      for (int j = 1; j <= size_z-2; j++) {
         f(i,j) = 
            u(i,j)*(helm_parameter + 
                  ux(i,j)*Dx(i,1) + uz(i,j)*Dz(j,1) +
                  uxx(i,j)*Dxx(i,1) + uzz(i,j)*Dzz(j,1) +
                  uxz(i,j)*Dx(i,1)*Dz(j,1)) +
            u(i-1,j)*(
                  ux(i,j)*Dx(i,0) + uxx(i,j)*Dxx(i,0) +
                  uxz(i,j)*Dx(i,0)*Dz(j,1)) +
            uin_right(j)*(
                  ux(i,j)*Dx(i,2) + uxx(i,j)*Dxx(i,2) +
                  uxz(i,j)*Dx(i,2)*Dz(j,1)) +
            u(i,j-1)*(
                  uz(i,j)*Dz(j,0) + uzz(i,j)*Dzz(j,0) +
                  uxz(i,j)*Dx(i,1)*Dz(j,0)) +
            u(i,j+1)*(
                  uz(i,j)*Dz(j,2) + uzz(i,j)*Dzz(j,2) +
                  uxz(i,j)*Dx(i,1)*Dz(j,2)) +
            uxz(i,j)*(u(i-1,j-1)*Dx(i,0)*Dz(j,0) +
                  uin_right(j-1)*Dx(i,2)*Dz(j,0) +
                  u(i-1,j+1)*Dx(i,0)*Dz(j,2) +
                  uin_right(j+1)*Dx(i,2)*Dz(j,2));
//         fprintf(stderr,"%d: %f %f %f\n",j,u(i-1,j),u(i,j),uin_right(j));
//         fprintf(stderr,"   [%f %f %f]=%f\n",uxx(i,j)*Dxx(i,0),uxx(i,j)*Dxx(i,1),uxx(i,j)*Dxx(i,2),f(i,j));
      }
      int j = 0;
      f(i,j) = u(i,j)*(u_bot(i) + ux_bot(i)*Dx(i,1) + uz_bot(i)*Dz(j,1)) +
            u(i-1,j)*ux_bot(i)*Dx(i,0) + uin_right(j)*ux_bot(i)*Dx(i,2) +
            u(i,j+1)*uz_bot(i)*Dz(j,2);
//      fprintf(stderr,"Bottom %d %gL %gH %gR %gU = %g\n",i,u(i-1,j),u(i,j),uin_right(j),u(i,j+1),f(i,j));
      j=size_z-1;
      f(i,j) = u(i,j)*(u_top(i) + ux_top(i)*Dx(i,1) + uz_top(i)*Dz(j,1)) +
            u(i-1,j)*ux_top(i)*Dx(i,0) + uin_right(j)*ux_top(i)*Dx(i,2) +
            u(i,j-1)*uz_top(i)*Dz(j,0);
   }

   /* Left and right BC's */
   if (symmetry_type == SYM_NONE) {
      if (local_x_lbound == 0) {
         int i = 0;
         for (int j = 1; j <= size_z-2; j++) {
            f(i,j) = u(i,j)*(u_left(j) + ux_left(j)*Dx(i,1) + uz_left(j)*Dz(j,1)) +
               u(i,j-1)*uz_left(j)*Dz(j,0) + u(i,j+1)*uz_left(j)*Dz(j,2) +
               u(i+1,j)*ux_left(j)*Dx(i,2);
         }
         int j=0;
         /* The corner points are special.  With a finite difference operator
            and typical "normal" boundary conditions (no tangential derivative),
            the corner points don't actually enter into the calculation anywhere.
            However, if this FD operator arises from a transformed grid, the corner
            points will affect things through the Jacobian, which will result in
            tangential derivatives.

            So, we have a choice -- we can consider the corner points to belong
            to the top/bottom or left/right.  Since we're stereotypically going
            to have solid boundaries at top/bottom, that's where we'll take
            it -- that way a viscous problem has u=0 at the corners */
         f(i,j) = u(i,j)*(u_bot(i) + ux_bot(i)*Dx(i,1) + uz_bot(i)*Dz(j,1)) +
            u(i,j+1)*uz_bot(i)*Dz(j,2) + u(i+1,j)*ux_bot(i)*Dx(i,2);
         j=size_z-1;
         f(i,j) = u(i,j)*(u_top(i) + ux_top(i)*Dx(i,1) + uz_top(i)*Dz(j,1)) +
            u(i,j-1)*uz_top(i)*Dz(j,0) + u(i+1,j)*ux_top(i)*Dx(i,2);

      } 
      if (local_x_ubound == size_x-1) {
         int i = size_x-1;

         for (int j = 1; j <= size_z-2; j++) {
            f(i,j) = u(i,j)*(u_right(j) + ux_right(j)*Dx(i,1) + uz_right(j)*Dz(j,1)) +
               u(i,j-1)*uz_right(j)*Dz(j,0) + u(i,j+1)*uz_right(j)*Dz(j,2) +
               u(i-1,j)*ux_right(j)*Dx(i,0);
         }
         int j=0;
         f(i,j) = u(i,j)*(u_bot(i) + ux_bot(i)*Dx(i,1) + uz_bot(i)*Dz(j,1)) +
            u(i,j+1)*uz_bot(i)*Dz(j,2) + u(i-1,j)*ux_bot(i)*Dx(i,0);
         j=size_z-1;
         f(i,j) = u(i,j)*(u_top(i) + ux_top(i)*Dx(i,1) + uz_top(i)*Dz(j,1)) +
            u(i,j-1)*uz_top(i)*Dz(j,0) + u(i-1,j)*ux_top(i)*Dx(i,0);
      }
   }
}


/* Apply one level of red-black relaxation on the given residual vector f,
   storing the result in x */
void MG_Solver::do_redblack(blitz::Array<double,2> & f, blitz::Array<double,2> & u) {
   assert(u.stride(secondDim) == 1);
   /* The objective here will be to build and solve many 1D operators (in z),
      in two phases.  The first will deal with the "red" lines, which are solved
      in isolation like Jacobi iteration.  Then, the influence of these points on
      the "black" lines gets computed and the "black" lines are solved.

      With our parallel split, all processors except possibly the last will
      have an even number of lines.  Arbitrarily assigning the even lines to red
      (starting at 0) gives a distribution of something like:

      RBRBRBRB || RBRBRB || RBRBRB || RBRBRBR

      Parallel communication here is limited to just one neighbour -- the locally-
      beginning red line will get sent to the left for the computation of its
      rightmost black line (wrapping around if the x-boundary is periodic with
      an even number of points).

      This is low-hanging fruit for simultaneous communication and computation. */
   Array<double,1> ul_right(size_z), // Array for receiving right neighbour
               u_coef(size_z), // coefficient for the u term
               uz_coef(size_z), // ux_term
               uzz_coef(size_z),
               f_line(size_z),
               u_line(size_z);
   double bc_bot, bc_z_bot, bc_top, bc_z_top;
   ul_right = -999;

   // MPI_Request objects for sending and receiving, for nonblocking
   // send and receive operations
   MPI_Request send_req, rec_req;
   MPI_Status ignoreme;

//   fprintf(stderr,"Redblack: %d/%d, %d local\n",myrank,nproc,u.extent(firstDim));
//   SYNC(cerr << f; cerr.flush());
//   MPI_Finalize(); exit(1);

   int right_neighbour;
   int left_neighbour;
   /* Figure out which our left and right neighbours are.  This is obvious,
      except at the left and right boundaries.  If there's a true BC or non-periodic
      symmetry, we have no data to transfer.  If there is periodic symmetry, we have
      data to transfer provided that there's also an even number of points in x;
      otherwise the left and right boundaries are both "red", and thus handled
      independently. */
   if (myrank != nproc-1) {
      right_neighbour = myrank+1;
   } else if (symmetry_type == SYM_PERIODIC && (size_x % 2) == 0) {
      right_neighbour = 0;
   } else {
      right_neighbour = MPI_PROC_NULL;
   }

   if (myrank != 0) {
      left_neighbour = myrank - 1;
   }  else if (symmetry_type == SYM_PERIODIC && (size_x % 2) == 0) {
      left_neighbour = nproc - 1;
   } else {
      left_neighbour = MPI_PROC_NULL;
   }
   /* Set up the nonblocking call to receive data from the right neighbour */
   MPI_Irecv(ul_right.data(),size_z,MPI_DOUBLE,right_neighbour,
         MPI_ANY_TAG,my_comm,&rec_req);

   
   /* Solve the red lines */
   for (int i = local_x_lbound; i <= local_x_ubound; i+= 2) {
      // Coefficients for boudary conditions
//      fprintf(stderr,"Solving line %d\n",i);

      /* Since the terms are built on Kroneker products and we're only
         dealing with one line, there's only one coefficient of
         importance in Dx and Dxx -- the line-local one.  Copy those
         out here */
      double Dxh = Dx(i,1); // Dx here
      double Dxxh = Dxx(i,1); // Dxx here
      u_coef = 0; uz_coef = 0; uzz_coef = 0;

      /* If we're at the boundary and the problem has symmetry,
         then Dx and Dxx get modified to take that into account */
      if (i == 0 && symmetry_type != SYM_NONE) {
         if (symmetry_type == SYM_EVEN) {
            Dxh = Dxh + Dx(i,0);
            Dxxh = Dxxh + Dxx(i,0);
         } else if (symmetry_type == SYM_ODD) {
            Dxh = Dxh - Dx(i,0);
            Dxxh = Dxxh - Dxx(i,0);
         }
      }
      if (i == size_x-1 && symmetry_type != SYM_NONE) {
         if (symmetry_type == SYM_EVEN) {
            Dxh = Dxh + Dx(i,2);
            Dxxh = Dxxh + Dxx(i,2);
         } else if (symmetry_type == SYM_ODD) {
            Dxh = Dxh - Dx(i,2);
            Dxxh = Dxxh - Dxx(i,2);
         }
      }
      if (symmetry_type != SYM_NONE || 
            (i != 0 && i != size_x - 1)) { // Away from possible BC issues
         /* Store coefficients for u, uz, and uzz */
//         fprintf(stdout,"Interior line\n");
         for (int j = 1; j < size_z-1; j++) {
            // The uzz term can be copied directly over
            uzz_coef(j) = uzz(i,j); 

            // The uz term gets its direct contribution plus one from
            // the Dx*Dz term.
            uz_coef(j) = uz(i,j) + Dxh*uxz(i,j);
            
            // The u term gets the normal helmholtz parameter plus
            // the centre part of each of the Dx/Dxx terms
            u_coef(j) = helm_parameter+Dxxh*uxx(i,j)+Dxh*ux(i,j);
         }
         // Bottom boundary, Dirichlet term
         bc_bot = u_bot(i)+ux_bot(i)*Dxh;
         bc_top = u_top(i)+ux_top(i)*Dxh;
         // And Neumann terms
         bc_z_bot = uz_bot(i);
         bc_z_top = uz_top(i);
//         if (0&&i==4) {
//            fprintf(stderr,"RB line i=4i (%g/%g , %g/%g)\n",bc_bot,bc_top,bc_z_bot,bc_z_top);
//            cerr << u_coef << uz_coef << uzz_coef;
//         }
      } else if (i == 0  && symmetry_type == SYM_NONE) {
         /* Left boundary with no symmetry */
         /* As such, there is no uzz term */
//         fprintf(stderr,"Left, no symmetry\n");
         uzz_coef = 0;
         for (int j = 1; j < size_z-1; j++) {
            /* And the uz term is given from the left BC */
            uz_coef(j) = uz_left(j);
            /* And the u term is given from a combination
               of the ux and u BC terms */
            u_coef(j) = u_left(j) + ux_left(j)*Dx(i,1);
         }
//         cerr << u_left << ux_left << uz_left;
         /* Top/bottom BCs are the same as the interior */
         bc_bot = u_bot(i)+ux_bot(i)*Dx(i,1);
         bc_top = u_top(i)+ux_top(i)*Dx(i,1);
         bc_z_bot = uz_bot(i);
         bc_z_top = uz_top(i);
      } else if (i == size_x-1 && symmetry_type == SYM_NONE) {
//         fprintf(stdout,"Right, no symmetry\n");
         uzz_coef = 0;
         for (int j = 1; j < size_z-1; j++) {
            /* And the uz term is given from the left BC */
            uz_coef(j) = uz_right(j);
            /* And the u term is given from a combination
               of the ux and u BC terms */
            u_coef(j) = u_right(j) + ux_right(j)*Dx(i,1);
         }
         /* Top/bottom BCs are the same as the interior */
         bc_bot = u_bot(i)+ux_bot(i)*Dx(i,1);
         bc_top = u_top(i)+ux_top(i)*Dx(i,1);
         bc_z_bot = uz_bot(i);
         bc_z_top = uz_top(i);
      } else {
         fprintf(stderr,"Red line (%d) missing (%d->%d)!\n",i,local_x_lbound,local_x_ubound);
         abort();
      }

      f_line = f(i,Range::all()); // Copy the residual
//      if (i==4) {cerr << f_line; cerr.flush();}

      // Solve the line
//      fprintf(stderr,"Line (R) %d/%d\n",i,local_x_ubound);
      line_solve(u_line,f_line,uzz_coef,uz_coef,u_coef,
            bc_bot,bc_top,bc_z_bot,bc_z_top,Dz,Dzz);
//      if (i==4) {fprintf(stderr,"Line4(R):\n");cerr << u_line; cerr.flush();}
//      cerr << u_line;

      // Copy back into u
      u(i,Range::all()) = u_line;

      /* If we've just computed the local lbound, we want to send
         it to the left neighbour */
      if (i == local_x_lbound) {
//         fprintf(stderr,"%d is lbound on processor %d, sending.  First index %g\n",i,myrank,u(i,0));
         MPI_Isend(&u(local_x_lbound,0),size_z,MPI_DOUBLE,left_neighbour,
               0, my_comm,&send_req);
      }
   }
   // Now, black points, treating the last line specially
   for (int i = local_x_lbound+1; i <= local_x_ubound-1; i+= 2) {
      // Coefficients for boudary conditions
//      fprintf(stderr,"Solving line %d\n",i);

      /* Since the terms are built on Kroneker products and we're only
         dealing with one line, there's only one coefficient of
         importance in Dx and Dxx -- the line-local one.  Copy those
         out here */
      u_coef = 0; uz_coef = 0; uzz_coef = 0;
      
      f_line = f(i,Range::all()); // Copy the residual

      /* Store coefficients for u, uz, and uzz */
      for (int j = 1; j < size_z-1; j++) {
         uzz_coef(j) = uzz(i,j); 
         uz_coef(j) = uz(i,j) + Dx(i,1)*uxz(i,j);
         u_coef(j) = helm_parameter+Dxx(i,1)*uxx(i,j)+Dx(i,1)*ux(i,j);

         // And modify the residual to take into account the
         // left and right neighbours, which we've already computed
         f_line(j) = f_line(j) -
               uxx(i,j)*(Dxx(i,0)*u(i-1,j) + Dxx(i,2)*u(i+1,j)) -
               ux(i,j)*(Dx(i,0)*u(i-1,j) + Dx(i,2)*u(i+1,j)) -
               uxz(i,j)*(Dx(i,0)*(Dz(j,0)*u(i-1,j-1)+Dz(j,2)*u(i-1,j+1)) +
                         Dx(i,2)*(Dz(j,0)*u(i+1,j-1)+Dz(j,2)*u(i+1,j+1)));
      }
      // Bottom boundary, Dirichlet term
      bc_bot = u_bot(i)+ux_bot(i)*Dx(i,1);
      bc_top = u_top(i)+ux_top(i)*Dx(i,1);
      // And Neumann terms
      bc_z_bot = uz_bot(i);
      bc_z_top = uz_top(i);

      // And adjust residual
      f_line(0) = f_line(0)-ux_bot(i)*(Dx(i,0)*u(i-1,0)+Dx(i,2)*u(i+1,0));
//      if (i==3) fprintf(stderr,"RX: %g*%g\n",Dx(i,2),u(i+1,0));
      f_line(size_z-1) = f_line(size_z-1) - ux_top(i)*
                        (Dx(i,0)*u(i-1,size_z-1)+Dx(i,2)*u(i+1,size_z-1));
//      if (i==3) cerr << f_line;

      // Solve the line
//      fprintf(stdout,"Line (B) %d/%d\n",i,local_x_ubound);
      line_solve(u_line,f_line,uzz_coef,uz_coef,u_coef,
            bc_bot,bc_top,bc_z_bot,bc_z_top,Dz,Dzz);

      // Copy back into u
      u(i,Range::all()) = u_line;
   }
   // finally, the local right-hand-point


   /* Make sure we've received the upper bound by now */
   MPI_Wait(&rec_req,&ignoreme);
//   for (int i = 0; i < nproc; i++)  {
//      if (i == myrank) cout << "Process " << myrank << " received right neighbour" << ul_right;
//      MPI_Barrier(my_comm);
//   }

   // If the upper bound is even, it's a red line and we've already done it
   if (local_x_ubound % 2 == 0) {
      // Make sure the send completed -- this should be a null op, but
      // it's nice to clear out the request object anyway
      MPI_Wait(&send_req,&ignoreme);
      return; 
   }

   int i = local_x_ubound;
   f_line = f(i,Range::all()); // Copy the residual
//   fprintf(stdout,"Solving line %d (ubound)\n",i);
   if (local_x_ubound == size_x-1 && symmetry_type != SYM_PERIODIC) {
      // Domain boundary
      if (symmetry_type == SYM_NONE) {
//         fprintf(stdout,"... with no symmetry\n");
         // BC
         uzz_coef = 0;
         for (int j = 1; j < size_z-1; j++) {
            /* And the uz term is given from the left BC */
            uz_coef(j) = uz_right(j);
            /* And the u term is given from a combination
               of the ux and u BC terms */
            u_coef(j) = u_right(j) + ux_right(j)*Dx(i,1);

            f_line(j) = f_line(j) - ux_right(j)*
                        (Dx(i,0)*u(i-1,j));
         }
         /* Top/bottom BCs are the same as the interior */
         bc_bot = u_bot(i)+ux_bot(i)*Dx(i,1);
         f_line(0) = f_line(0) - ux_bot(i)*Dx(i,0)*u(i-1,0);
         bc_top = u_top(i)+ux_top(i)*Dx(i,1);
         f_line(size_z-1) = f_line(size_z-1) - 
                            ux_top(i)*Dx(i,0)*u(i-1,size_z-1);
         bc_z_bot = uz_bot(i);
         bc_z_top = uz_top(i);
      }
      else { // Even or odd symmetry
         int SYM = (symmetry_type == SYM_EVEN ? 1 : -1);
//         fprintf(stderr,"... with even/odd symmetry\n");
         for (int j = 1; j < size_z-1; j++) {
            uzz_coef(j) = uzz(i,j); 
            uz_coef(j) = uz(i,j) + (Dx(i,1)+SYM*Dx(i,2))*uxz(i,j);
            u_coef(j) = helm_parameter+
                        (Dxx(i,1)+SYM*Dxx(i,2))*uxx(i,j)+
                        (Dx(i,1)+SYM*Dx(i,2))*ux(i,j);

            // And modify the residual to take into account the
            // left and right neighbours, which we've already computed
            f_line(j) = f_line(j) -
                  uxx(i,j)*(Dxx(i,0)*u(i-1,j)) -
                  ux(i,j)*(Dx(i,0)*u(i-1,j)) -
                  uxz(i,j)*(Dx(i,0)*(Dz(j,0)*u(i-1,j-1)+Dz(j,2)*u(i-1,j+1)));
         }
         /* Top/bottom BCs are the same as the interior */
         bc_bot = u_bot(i)+ux_bot(i)*(Dx(i,1)+SYM*Dx(i,2));
         f_line(0) = f_line(0) - ux_bot(i)*Dx(i,0)*u(i-1,0);
         bc_top = u_top(i)+ux_top(i)*(Dx(i,1)+SYM*Dx(i,2));
         f_line(size_z-1) = f_line(size_z-1) - 
                            ux_top(i)*Dx(i,0)*u(i-1,size_z-1);
         bc_z_bot = uz_bot(i);
         bc_z_top = uz_top(i);
      }
   } else {
//      fprintf(stdout,"... as an interior point\n");
      /* Interior point or periodic symmetry */
      for (int j = 1; j < size_z-1; j++) {
         uzz_coef(j) = uzz(i,j); 
         uz_coef(j) = uz(i,j) + Dx(i,1)*uxz(i,j);
         u_coef(j) = helm_parameter+Dxx(i,1)*uxx(i,j)+Dx(i,1)*ux(i,j);

         // And modify the residual to take into account the
         // left and right neighbours, which we've already computed
         f_line(j) = f_line(j) -
               uxx(i,j)*(Dxx(i,0)*u(i-1,j) + Dxx(i,2)*ul_right(j)) -
               ux(i,j)*(Dx(i,0)*u(i-1,j) + Dx(i,2)*ul_right(j)) -
               uxz(i,j)*(Dx(i,0)*(Dz(j,0)*u(i-1,j-1)+Dz(j,2)*u(i-1,j+1)) +
                         Dx(i,2)*(Dz(j,0)*ul_right(j-1)+Dz(j,2)*ul_right(j+1)));
      }
      // Bottom boundary, Dirichlet term
      bc_bot = u_bot(i)+ux_bot(i)*Dx(i,1);
      bc_top = u_top(i)+ux_top(i)*Dx(i,1);
      // And Neumann terms
      bc_z_bot = uz_bot(i);
      bc_z_top = uz_top(i);

      // And adjust residual
      f_line(0) = f_line(0)-ux_bot(i)*(Dx(i,0)*u(i-1,0)+Dx(i,2)*ul_right(0));
//      if (i==3) fprintf(stderr,"RX: %g*%g\n",Dx(i,2),ul_right(0));
      f_line(size_z-1) = f_line(size_z-1) - ux_top(i)*
                        (Dx(i,0)*u(i-1,size_z-1)+Dx(i,2)*ul_right(size_z-1));
//      if (i==3) cerr << f_line;
   }
   // Solve the line
//   cerr << uzz_coef << uz_coef << u_coef;
      
   line_solve(u_line,f_line,uzz_coef,uz_coef,u_coef,
         bc_bot,bc_top,bc_z_bot,bc_z_top,Dz,Dzz);
//      if (i==3) cerr << u_line;

   // Copy back into u
   u(i,Range::all()) = u_line;

   // Make sure the send completed
   MPI_Wait(&send_req,&ignoreme);
//   fprintf(stdout,"Exiting Redblack %d/%d\n",myrank,nproc);
}

void MG_Solver::coarsen_line(blitz::Array<double,1> & q, bool even) {
   /* Coarsen a 1D line, storing the result in local_coarse_1d */

   /* The coarsening operator is full weighting, -without- modification for
      an uneven grid.  This means that each point in the resulting coarse grid
      is given by:
         0.25 * left + 0.5 * self + 0.25 * right.
      The left and right boundaries, if they are included, are fixed.  If
      there is a "kept interval" (from being unable to divide the interior
      intervals in half), then we still use the same formula as above.  The
      conceit is that the line is low-pass filtered and -then- subsampled.

      Periodic "boundaries" get handled via a modification of the above formula;
      the "boundray" is a phantom point, with its value being 0 (odd), nearest-
      neighbour (even), or 0.5*(nearest_neighbour+opposite_neighbour) (periodic).
      This affects the weights of the above formula. */
   double q_left, q_right; // Neighbours

//   if (myrank == 0) {
//      cout << "Coarsening line:\n";
//   }
//   for (int i = 0; i < nproc; i++) {
//      if (i == myrank) cout << q;
//      MPI_Barrier(my_comm);
//   }

   // Send left neighbour, receive right
   MPI_Status ignoreme[2];
   int lefty = myrank - 1; int righty = myrank + 1;
   if (lefty < 0) {lefty = nproc-1;};
   if (righty >= nproc) {righty = 0;};
   if (symmetry_type != SYM_PERIODIC) {
      /* Have some kind of closed boundary, so we don't transfer a point around
         0/nproc-1 */
      if (myrank == 0) {lefty = MPI_PROC_NULL;};
      if (myrank == nproc-1) {righty = MPI_PROC_NULL;};
   }
   MPI_Request right_receive[2];
   MPI_Irecv(&q_right,1,MPI_DOUBLE,righty,0,my_comm,&right_receive[0]);
   MPI_Isend(&q(local_x_lbound),1,MPI_DOUBLE,lefty,0,my_comm,&right_receive[1]);
//   MPI_Sendrecv(&q(local_x_lbound),1,MPI_DOUBLE,lefty,0,
//         &q_right,1,MPI_DOUBLE,righty,0,my_comm,&ignoreme);
   // Send right neighbour, receive left
   MPI_Request left_receive[2];
   MPI_Irecv(&q_left,1,MPI_DOUBLE,lefty,0,my_comm,&left_receive[0]);
   MPI_Isend(&q(local_x_ubound),1,MPI_DOUBLE,righty,0,my_comm,&left_receive[1]);
//   MPI_Sendrecv(&q(local_x_ubound),1,MPI_DOUBLE,right,0,
//         &q_left,1,MPI_DOUBLE,lefty,0,my_comm,&ignoreme);
   
   // Now, loop over the local_coarse_1d
   for (int i = local_coarse_1d.lbound(firstDim)+1;
          i <= local_coarse_1d.ubound(firstDim)-1; i++) {
      int here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      int left = here-1; int right = here+1;
      local_coarse_1d(i) = 0.25*q(left) + 0.5*q(here) + 0.25*q(right);
   }
   if (local_coarse_1d.ubound(firstDim) > local_coarse_1d.lbound(firstDim)) {
      // If there is more than one point in the local coarse representation
      MPI_Waitall(2,right_receive,ignoreme); // Get the right neighbour
      int i = local_coarse_1d.ubound(firstDim);
      int here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      int left = here-1; int right = here + 1;
      if (right <= local_x_ubound) q_right = q(right);
      if (here == size_x-1 && symmetry_type == SYM_NONE) {
         /* If this point represents the right bound, copy the boundary
            value over directly */
         local_coarse_1d(i) = q(here);
      } else if (here != size_x - 1 || symmetry_type == SYM_PERIODIC) {
         // If we're not at the right-bound of the array -or- there is
         // periodic symmetry, use the interior formula
         local_coarse_1d(i) = 0.25*q(left) + 0.5*q(here) + 0.25*q_right;
      } else if (here == size_x - 1 && (even || symmetry_type == SYM_EVEN)) {
         // Even extension, so the phantom right point is equal to the current
         // point
         local_coarse_1d(i) = 0.25*q(left) + 0.75*q(here);
      } else if (here == size_x - 1 && !even && symmetry_type == SYM_ODD) {
         // As above, but the right point is negtaive current
         local_coarse_1d(i) = 0.25*q(left) + 0.25*q(here);
      }
      MPI_Waitall(2,left_receive,ignoreme); // And the left neighbour
      i = local_coarse_1d.lbound(firstDim);
      here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      left = here-1; right = here + 1;
      if (left >= local_x_lbound) q_left = q(left);
      if (here == 0 && symmetry_type == SYM_NONE) {
         // Left boundary, with BC
         local_coarse_1d(i) = q(here);
      } else if (here != 0 || symmetry_type == SYM_PERIODIC) {
         local_coarse_1d(i) = 0.25*q_left + 0.5*q(here) + 0.25*q(right);
      } else if (here == 0 && (even || symmetry_type == SYM_EVEN)) {
         local_coarse_1d(i) = 0.25*q(right) + 0.75*q(here);
      } else if (here == 0 && !even && symmetry_type == SYM_ODD) {
         // As above, but the right point is negtaive current
         local_coarse_1d(i) = 0.25*q(right) + 0.25*q(here);
      }

   } else { 
      // Only one point in local coarse grid
      MPI_Waitall(2,right_receive,ignoreme); // Get the right neighbour
      MPI_Waitall(2,left_receive,ignoreme); // And the left neighbour
      int i = local_coarse_1d.lbound(firstDim);
      int here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      int left = here-1; int right = here + 1;
      if (left >= local_x_lbound) q_left = q(left);
      if (right <= local_x_ubound) q_right = q(right);
      if ((here != 0 && here != size_x - 1) || symmetry_type == SYM_PERIODIC) {
         // Interior point
         local_coarse_1d(i) = 0.25*q_left + 0.5*q(here) + 0.25*q_right;
      } else if (((here == 0) || (here== size_x-1)) && symmetry_type == SYM_NONE) {
         // Boundary point with BC
         local_coarse_1d(i) = q(here);
      } else if (here == 0) {
         if (symmetry_type == SYM_EVEN || even) {
            local_coarse_1d(i) = 0.25*q_right+0.75*q(here);
         } else {
            assert(symmetry_type == SYM_ODD);
            local_coarse_1d(i) = 0.25*q_right+0.25*q(here);
         }
      } else if (here == size_x-1) {
         if (symmetry_type == SYM_EVEN || even) {
            local_coarse_1d(i) = 0.25*q_left+0.75*q(here);
         } else {
            assert(symmetry_type == SYM_ODD);
            local_coarse_1d(i) = 0.25*q_left+0.25*q(here);
         }
      }
   }
//   for (int i = 0; i < nproc; i++) {
//      if (myrank == i) cout << local_coarse_1d;
//      MPI_Barrier(my_comm);
//   }
}
void MG_Solver::coarsen_grid(blitz::Array<double,2> & q, bool even) {
   /* Coarsen a 2D line, storing the result in local_coarse_2d */

   /* The coarsening operator is full weighting, -without- modification for
      an uneven grid.  This means that each point in the resulting coarse grid
      is given by:
         0.25 * left + 0.5 * self + 0.25 * right.
      The left and right boundaries, if they are included, are fixed.  If
      there is a "kept interval" (from being unable to divide the interior
      intervals in half), then we still use the same formula as above.  The
      conceit is that the line is low-pass filtered and -then- subsampled.

      Periodic "boundaries" get handled via a modification of the above formula;
      the "boundray" is a phantom point, with its value being 0 (odd), nearest-
      neighbour (even), or 0.5*(nearest_neighbour+opposite_neighbour) (periodic).
      This affects the weights of the above formula. */
   Array<double,1> q_left(size_z), q_right(size_z); // Neighbours
   Range all = Range::all();
   q_left = -999; q_right = -999;

//   if (myrank == 0) {
//      cout << "Coarsening grid:\n";
//   }
//   for (int i = 0; i < nproc; i++) {
//      if (i == myrank) cout << q;
//      MPI_Barrier(my_comm);
//   }

   // Send left neighbour, receive right
   MPI_Status ignoreme[2];
   int lefty = myrank - 1; int righty = myrank + 1;
   if (lefty < 0) {lefty = nproc-1;};
   if (righty >= nproc) {righty = 0;};
//   fprintf(stderr,"Communication processors: %d %d\n",lefty,righty);
   if (symmetry_type != SYM_PERIODIC) {
      /* Have some kind of closed boundary, so we don't transfer a point around
         0/nproc-1 */
      if (myrank == 0) {lefty = MPI_PROC_NULL;};
      if (myrank == nproc-1) {righty = MPI_PROC_NULL;};
   }
   // Send left neighbour, receive right
   MPI_Request right_receive[2];
   MPI_Irecv(&q_right(0),size_z,MPI_DOUBLE,righty,0,my_comm,&right_receive[0]);
   MPI_Isend(&q(local_x_lbound,0),size_z,MPI_DOUBLE,lefty,0,my_comm,&right_receive[1]);
   // Send right neighbour, receive left
   MPI_Request left_receive[2];
   MPI_Irecv(&q_left(0),size_z,MPI_DOUBLE,lefty,0,my_comm,&left_receive[0]);
   MPI_Isend(&q(local_x_ubound,0),size_z,MPI_DOUBLE,righty,0,my_comm,&left_receive[1]);
//   MPI_Sendrecv(&q(local_x_lbound),1,MPI_DOUBLE,lefty,0,
//         &q_right,1,MPI_DOUBLE,righty,0,my_comm,&ignoreme);
//      fprintf(stderr,"left_receive status: %d %d\n",ignoreme[0],ignoreme[1]);
//      q_left(all) = q(local_x_ubound,all);
//   MPI_Sendrecv(&q(local_x_ubound),1,MPI_DOUBLE,right,0,
//         &q_left,1,MPI_DOUBLE,lefty,0,my_comm,&ignoreme);
   
   // Now, loop over the local_coarse_2d
   for (int i = local_coarse_2d.lbound(firstDim)+1;
          i <= local_coarse_2d.ubound(firstDim)-1; i++) {
      int here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      int left = here-1; int right = here+1;
      local_coarse_2d(i,all) = 0.25*q(left,all) + 0.5*q(here,all) + 0.25*q(right,all);
   }
   if (local_coarse_2d.ubound(firstDim) > local_coarse_2d.lbound(firstDim)) {
      // If there is more than one point in the local coarse representation
      MPI_Waitall(2,right_receive,ignoreme); // Get the right neighbour
      int i = local_coarse_2d.ubound(firstDim);
      int here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      int left = here-1; int right = here + 1;
      // If the rightmost point we use in our local coarsening is itself
      // in the set of points we have, we don't actually need q_right
      // sent from our neighbour.
      if (right <= local_x_ubound) q_right = q(right,all);
      if (here == size_x-1 && symmetry_type == SYM_NONE) {
         /* If this point represents the right bound, copy the boundary
            value over directly */
         local_coarse_2d(i,all) = q(here,all);
      } else if (here != size_x - 1 || symmetry_type == SYM_PERIODIC) {
         // If we're not at the right-bound of the array -or- there is
         // periodic symmetry, use the interior formula
         local_coarse_2d(i,all) = 0.25*q(left,all) + 0.5*q(here,all) + 0.25*q_right(all);
      } else if (here == size_x - 1 && (even || symmetry_type == SYM_EVEN)) {
         // Even extension, so the phantom right point is equal to the current
         // point
         local_coarse_2d(i,all) = 0.25*q(left,all) + 0.75*q(here,all);
      } else if (here == size_x - 1 && !even && symmetry_type == SYM_ODD) {
         // As above, but the right point is negtaive current
         local_coarse_2d(i,all) = 0.25*q(left,all) + 0.25*q(here,all);
      }
      MPI_Waitall(2,left_receive,ignoreme); // And the left neighbour
      i = local_coarse_2d.lbound(firstDim);
      here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      left = here-1; right = here + 1;
      if (left >= local_x_lbound) {
//         fprintf(stderr,"Resetting q_left\n");
         q_left = q(left,all);
      }
      if (here == 0 && symmetry_type == SYM_NONE) {
         // Left boundary, with BC
         local_coarse_2d(i,all) = q(here,all);
      } else if (here != 0 || symmetry_type == SYM_PERIODIC) {
//         fprintf(stdout,"(%d) Kept Interval: %d\n",myrank,kept_interval);
//         fprintf(stdout,"%d %d %d\n",i,here,right);
//         fprintf(stderr,"Not left boundary or left and periodic symmetry\n");
//         cerr << q_left;
         local_coarse_2d(i,all) = 0.25*q_left + 0.5*q(here,all) + 0.25*q(right,all);
      } else if (here == 0 && (even || symmetry_type == SYM_EVEN)) {
         local_coarse_2d(i,all) = 0.25*q(right,all) + 0.75*q(here,all);
      } else if (here == 0 && !even && symmetry_type == SYM_ODD) {
         // As above, but the right point is negtaive current
         local_coarse_2d(i,all) = 0.25*q(right,all) + 0.25*q(here,all);
      }

   } else { 
      // Only one point in local coarse grid
      MPI_Waitall(2,right_receive,ignoreme); // Get the right neighbour
      MPI_Waitall(2,left_receive,ignoreme); // And the left neighbour
      int i = local_coarse_2d.lbound(firstDim);
      int here = 2*i;
      if ((kept_interval >= 0) && here > kept_interval) here--;
      int left = here-1; int right = here + 1;
      if (left >= local_x_lbound) q_left = q(left,all);
      if (right <= local_x_ubound) q_right = q(right,all);
      if (here != 0 && here != size_x - 1 || symmetry_type == SYM_PERIODIC) {
         // Interior point
         local_coarse_2d(i,all) = 0.25*q_left + 0.5*q(here,all) + 0.25*q_right;
      } else if (((here == 0) || (here== size_x-1)) && symmetry_type == SYM_NONE) {
         // Boundary point with BC
         local_coarse_2d(i,all) = q(here,all);
      } else if (here == 0) {
         if (symmetry_type == SYM_EVEN || even) {
            local_coarse_2d(i,all) = 0.25*q_right+0.75*q(here,all);
         } else {
            assert(symmetry_type == SYM_ODD);
            local_coarse_2d(i,all) = 0.25*q_right+0.25*q(here,all);
         }
      } else if (here == size_x-1) {
         if (symmetry_type == SYM_EVEN || even) {
            local_coarse_2d(i,all) = 0.25*q_left+0.75*q(here,all);
         } else {
            assert(symmetry_type == SYM_ODD);
            local_coarse_2d(i,all) = 0.25*q_left+0.25*q(here,all);
         }
      }
   }
//   for (int i = 0; i < nproc; i++) {
//      if (myrank == i) cout << local_coarse_2d;
//      MPI_Barrier(my_comm);
//   }
}

void MG_Solver::interpolate_grid(Array<double,2> & q) {
   q = -999;
   Range all = Range::all();
   Array<double,1> qc_left(size_z), qc_right(size_z);
   qc_left = -999; qc_right = -999;
//   if (myrank == 0) {
//      cout << "Interpolating grid:\n";
//   }
//   for (int i = 0; i < nproc; i++) {
//      if (i == myrank) cout << local_coarse_2d;
//      MPI_Barrier(my_comm);
//   }

   // This entire process uses more communication than may be
   // necessary.  If a coarse point is actually on the local
   // fine-grid boundary, then it is copied straight over, so
   // the left/right neighbour may not be needed.  However,
   // -which- neighbours are needed is a bit tricky to define since
   // it depends on the local distribution, where the kept interval
   // is (if any), and the periodicity of the problem.
   // Send left neighbour, receive right
   MPI_Status ignoreme[2];
   int lefty = myrank - 1; int righty = myrank + 1;
   int lbound = local_coarse_2d.lbound(firstDim);
   int ubound = local_coarse_2d.ubound(firstDim);
   if (lefty < 0) {lefty = nproc-1;};
   if (righty >= nproc) {righty = 0;};
   if (symmetry_type != SYM_PERIODIC) {
      /* Have some kind of closed boundary, so we don't transfer a point around
         0/nproc-1 */
      if (myrank == 0) {lefty = MPI_PROC_NULL;};
      if (myrank == nproc-1) {righty = MPI_PROC_NULL;};
   }
   MPI_Request right_receive[2];
   MPI_Irecv(&qc_right(0),size_z,MPI_DOUBLE,righty,0,my_comm,&right_receive[0]);
   MPI_Isend(&local_coarse_2d(lbound,0),size_z,MPI_DOUBLE,lefty,0,my_comm,&right_receive[1]);
//   MPI_Sendrecv(&q(local_x_lbound),1,MPI_DOUBLE,lefty,0,
//         &q_right,1,MPI_DOUBLE,righty,0,my_comm,&ignoreme);
   // Send right neighbour, receive left
   MPI_Request left_receive[2];
   MPI_Irecv(&qc_left(0),size_z,MPI_DOUBLE,lefty,0,my_comm,&left_receive[0]);
   MPI_Isend(&local_coarse_2d(ubound,0),size_z,MPI_DOUBLE,righty,0,my_comm,&left_receive[1]);
   /* Loop over interior lines, which we can definitely do without communication */
//   fprintf(stderr,"Kept interval is %d\n",kept_interval);
   for (int i = local_x_lbound+1; i <= local_x_ubound-1; i++) {
      if ((symmetry_type != SYM_NONE && ((i+(kept_interval >= 0 && i > kept_interval))%2 == 0)) ||
            (symmetry_type == SYM_NONE && (i+(kept_interval >= 0 && i > kept_interval))%2 == 0)) {
         // The above expression is a complicated way of saying that
         // we're exactly at a coarse-grid point
         int here;
//         if (symmetry_type != SYM_NONE)
//            here = (i+(kept_interval>=0 && i >= kept_interval)+(symmetry_type != SYM_NONE))/2;
//         else  
            here = (i+(kept_interval>=0 && i > kept_interval))/2;
//         fprintf(stderr,"%d corresponds exactly to coarse %d\n",i,here);
         q(i,all) = local_coarse_2d(here,all);
      } else {
         // We're split between points, so we have to average the left and right
         // coarse values
         int left;
//         if (symmetry_type != SYM_NONE)
//            left = (i+(kept_interval>=0 && i >= kept_interval)+(symmetry_type != SYM_NONE))/2;
//         else  
            left = (i+(kept_interval>=0 && i > kept_interval))/2;
         int right = left+1;
//         fprintf(stderr,"%d is between coarse %d and %d\n",i,left,right);
         q(i,all) = 0.5*(local_coarse_2d(left,all)+local_coarse_2d(right,all));
      }
   }
   // Handle the right point
   MPI_Waitall(2,right_receive,ignoreme); // Grab the right neighbour
   {
      int i = local_x_ubound;
      if ((symmetry_type != SYM_NONE && ((i+(kept_interval >= 0 && i > kept_interval))%2 == 0)) ||
            (symmetry_type == SYM_NONE && (i+(kept_interval >= 0 && i > kept_interval))%2 == 0)) {
                  
         // The ubound can itself be a coarse point.  It will be in the case of a solid
         // boundary.
         int here;
         if (symmetry_type != SYM_NONE)
            here = (i+(kept_interval>=0 && i >= kept_interval)+(symmetry_type != SYM_NONE))/2;
         else  
            here = (i+(kept_interval>=0 && i > kept_interval))/2;
//         fprintf(stderr,"Boundary: %d is exactly coarse %d\n",i,here);
         q(i,all) = local_coarse_2d(here,all);
      } else {
         if (i == size_x-1 && symmetry_type == SYM_EVEN) {
            // Even symmetry, so copy the first coarse value
//            fprintf(stderr,"Boundary: %d is copying %d\n",i,local_coarse_2d.ubound(firstDim));
           q(i,all) = local_coarse_2d(local_coarse_2d.ubound(firstDim),all);
         } else if (i == size_x-1 && symmetry_type == SYM_ODD) {
//            fprintf(stderr,"Boudnary: %d is copying %d\n",i,local_coarse_2d.ubound(firstDim));
           // Odd symmetry is zero at the boundary, so here use half the coarse value
           q(i,all) = 0.5*local_coarse_2d(local_coarse_2d.ubound(firstDim),all);
         } else if (i == size_x-1 && symmetry_type == SYM_PERIODIC) {
          q(i,all) = 0.5*local_coarse_2d(local_coarse_2d.ubound(firstDim),all) + 0.5*qc_right; 
         } else {
            assert(i != 0);
            int left;
//            if (symmetry_type != SYM_NONE)
//               left = (i+(kept_interval>=0 && i >= kept_interval)-(symmetry_type != SYM_NONE))/2;
//            else  
               left = (i+(kept_interval>=0 && i > kept_interval)-(symmetry_type != SYM_NONE))/2;
            q(i,all) = 0.5*(local_coarse_2d(left,all)+qc_right);
         }
      }
   }
   // Now, handle the left point
   MPI_Waitall(2,left_receive,ignoreme); // Grab the left neighbour
   {
      int i = local_x_lbound;
      if ((symmetry_type != SYM_NONE && ((i+(kept_interval >= 0 && i > kept_interval))%2 == 0)) ||
            (symmetry_type == SYM_NONE && (i+(kept_interval >= 0 && i > kept_interval))%2 == 0)) {
         // The lbound can itself be a coarse point.  It will be in the case of a solid
         // boundary.
         int here;
         if (symmetry_type != SYM_NONE)
            here = (i+(kept_interval>=0 && i >= kept_interval)+(symmetry_type != SYM_NONE))/2;
         else  
            here = (i+(kept_interval>=0 && i > kept_interval))/2;
//         fprintf(stderr,"Boundary: %d is exactly coarse %d\n",i,here);
         q(i,all) = local_coarse_2d(here,all);
      } else {
         if (i == 0 && symmetry_type == SYM_EVEN) {
            // Even symmetry, so copy the first coarse value
//            fprintf(stderr,"Even boundary: %d copies 0\n",i);
           q(i,all) = local_coarse_2d(0,all);
         } else if (i == 0 && symmetry_type == SYM_ODD) {
           // Odd symmetry is zero at the boundary, so here use half the coarse value
//            fprintf(stderr,"Odd boundary: %d is half 0\n",i);
           q(i,all) = 0.5*local_coarse_2d(0,all);
         } else if (i == 0 && symmetry_type == SYM_PERIODIC) {
//            fprintf(stderr,"Perioidic boundary: %d is weight of 0 and left\n",i);
           q(i,all) = 0.5*local_coarse_2d(0,all) + 0.5*qc_left; 
         } else {
            assert(i != 0);
            int right;
//            if (symmetry_type != SYM_NONE)
//               right = (i+(kept_interval>=0 && i >= kept_interval)+(symmetry_type != SYM_NONE))/2+1;
//            else  
               right = (i+(kept_interval>=0 && i > kept_interval))/2+1;
//            fprintf(stderr,"Internal boundary: %d is between left and %d\n",i);
            q(i,all) = 0.5*(local_coarse_2d(right,all)+qc_left);
         }
      }
   }

//   if (myrank == 0) {
//      cout << "Interpolated grid:\n";
//   }
//   for (int i = 0; i < nproc; i++) {
//      if (i == myrank) cout << q;
//      MPI_Barrier(my_comm);
//   }
      
}

void MG_Solver::cycle(CYCLE_TYPE cycle, Array<double,2> & f, Array<double,2> & u,
      double extra_in, double & extra_out,int pre, int mid, int post) {
   /* Load-balance f, solve using _cycle, and restore the proper output to u.
      Extra parameters for input and output are used in solving an indefinite
      problem, to match pointwise-mean and incompatible-f from the subspace.

      That is, for 
      Nabla u = f - extra_out; u_normal = 0,
      <u> = extra_in
   */

   //fprintf(stderr,"%g %g %g\n",f.data()[0],f.data()[1],f.data()[2]);
   if (any(f_balance.extent() == 0)) {
      TinyVector<int,2> base_vector; 
      base_vector(0) = local_x_lbound;
      base_vector(1) = 0;
      f_balance.resize(local_size_x,size_z);
      f_balance.reindexSelf(base_vector);
      u_balance.resize(local_size_x,size_z);
      u_balance.reindexSelf(base_vector);
   }

   if (any(f.ordering() != f_balance.ordering()) ||
       any(u.ordering() != u_balance.ordering())) {
      fprintf(stderr,"ERROR: Multigrid cycling uses standard C-ordered arrays\n");
      abort();
   }

   rebalance_array(f,f_balance,my_comm);

   _cycle(cycle,f_balance,u_balance,extra_in,extra_out,pre,mid,post);

   rebalance_array(u_balance,u,my_comm);
   // Extra_out is only valid on the first processor
   MPI_Bcast(&extra_out,1,MPI_DOUBLE,0,my_comm);
   return;
}

void MG_Solver::_cycle(CYCLE_TYPE cycle, Array<double,2> & f, Array<double,2> & u,
      double extra_in, double & extra_out,int pre, int mid, int post) {
   /* Perform a single #-cycle on the problem, with f as the residual RHS-vector */

   /* This allocation should be made only once. FIXME. */
   Array<double,2> defect(f.lbound(),f.extent()), correction(u.lbound(),u.extent());
   //fprintf(stderr,"_cycle: mean condition %g\n",extra_in);
   if (coarsest_level) {
      // Coarse-grid solve
      if (myrank != 0) return;
      if (!symbolic_factor || !numeric_factor ||
            !coarse_symbolic_ok || !coarse_numeric_ok) {
//         fprintf(stderr,"Building coarse operator\n");
         build_sparse_operator();
      }
      defect = f;
      u = 0;
      // Use UMFPACK solve
#if 1
      int sys = UMFPACK_At;
      double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
      //fprintf(stderr,"Coarse solve:\n");
      //cerr << f;
      double * u_ptr, *f_ptr;
      {
         /* If this is an indefinite problem, the coarse solve
            takes care of extra_in and extra_out, but that means
            that the problem is really of size (1+size_x*size_z);
            this doesn't actually fit in u or f.  So, allocate
            static blitz-vectors for the data, if necessary. */
         static blitz::Vector<double> extra_u(0), extra_f(0);
//         fprintf(stderr,"CG: Input residual\n");
//         cerr << f;
         if (indefinite_problem) {
//            fprintf(stderr,"Solving indefinite problem on coarse grid\n");
            if (extra_u.length() != (size_x*size_z+1))
               extra_u.resize(size_x*size_z+1);
            if (extra_f.length() != (size_x*size_z+1))
               extra_f.resize(size_x*size_z+1);
            // Copy the f-problem to extra_f
            
            memcpy(extra_f.data(),f.data(),size_x*size_z*sizeof(double));

//            cerr << f;
            extra_f(size_x*size_z) = extra_in;
            u_ptr = extra_u.data();
            f_ptr = extra_f.data();
//            fprintf(stderr,"CG: True residual\n");

//            cerr << extra_f << endl;

         } else {
            u_ptr = u.data();
            f_ptr = f.data();
         }
         int retval = umfpack_di_solve(sys,A_cols,A_rows,A_double,
                              u_ptr, f_ptr,
                              numeric_factor,
                              Control,Info);
         assert(!retval);
         if (indefinite_problem) {
//            fprintf(stderr,"CG: True u\n");
//            cerr << extra_u << endl;
            memcpy(u.data(),u_ptr,size_x*size_z*sizeof(double));
//            fprintf(stderr,"CG: Output U\n");
//            cerr << u;
            extra_out = extra_u(size_x*size_z);
//            fprintf(stderr,"With compatibility constant %g\n",extra_out);
         } else {
            extra_out = 0;
         }
         //fprintf(stderr,"CG: Output u\n");
         //cerr << u;
      }
//      fprintf(stderr,"Sparse solve [%d] mean %g vs %g\n",size_x,mean(u),extra_in);
//      fprintf(stderr,"Sparse solve returned %d\n",retval);
#else
      double fscale = max(abs(f));
      do {
         do_redblack(defect,correction);
         apply_operator(correction,f);
         u = u+correction;
         defect = defect-f;
      } while (max(abs(defect)) >= 1e-7*fscale);
#endif
//      cerr << u;
      return;
   }
   /* Need arrays for defect and correction */

   /* Perform one red-black relaxtion pass */
   //fprintf(stderr,"Pre-cyle one\n");
   if (pre >= 1) {
      do_redblack(f,correction);
   //cout << "New correction: " << correction;
   /* And apply the operator to find the defect */
      apply_operator(correction,defect);
      defect = -(defect - f);
      u = correction;
   /* Perform a second relaxation pass */
//   fprintf(stderr,"Pre-cycle two\n");
//      if (master()) fprintf(stderr,"RB precycle %d complete\n",0);
//      if (master()) fprintf(stderr,"U:\n");
//      SYNC(cerr << u; cerr.flush());
//      if (master()) fprintf(stderr,"Defect:\n");
//      SYNC(cerr << defect; cerr.flush());
   } else {
      defect = f;
      u = 0;
   }
   for (int i = 1; i < pre; i++) {
//      if (master()) fprintf(stderr,"Doing RB precycle %d\n",i);
      do_redblack(defect,correction);
//      if (master()) fprintf(stderr,"RB precycle %d complete\n",i);
//      if (master()) fprintf(stderr,"correction:\n");
//      SYNC(cerr << correction; cerr.flush());
      apply_operator(correction,f);
//      if (master()) fprintf(stderr,"f:\n");
//      SYNC(cerr << f; cerr.flush());
      u = u+correction;
      defect = defect-f;
//      if (master()) fprintf(stderr,"U:\n");
//      SYNC(cerr << u; cerr.flush());
//      if (master()) fprintf(stderr,"Defect:\n");
//      SYNC(cerr << defect; cerr.flush());
   }
//   MPI_Finalize(); exit(1);

   double coarse_extra_in=0;
   double coarse_extra_out=0;
//   if (1||indefinite_problem) {
//      /* extra_in will change, since part of the mean will already be accounted for */
//      double avg = pvsum(u,my_comm)/(size_x*size_z);
////      MPI_Allreduce(&avg,&avg,1,MPI_DOUBLE,MPI_SUM,my_comm);
//      coarse_extra_in = extra_in - avg;
//      fprintf(stderr,"Setting [//d] coarse_extra_in to %g given average %g and input %g\n",size_x,coarse_extra_in,avg,extra_in);
//   }

   /* Now, coarse-grid solve */
   switch (cycle) {
      case CYCLE_V:
//         if (master()) fprintf(stderr,"V[%d] local full defect\n",size_x);
//         SYNC(cerr << defect);
         coarsen_grid(defect);
         rebalance_array(local_coarse_2d,coarse_f,my_comm);
//         if (master()) fprintf(stderr,"V[%d] local coarse defect\n",size_x);
//         SYNC(cerr << coarse_f);
         if (coarse_solver) {
//            fprintf(stderr,"Coarse solve\n");
            coarse_solver->_cycle(CYCLE_V,coarse_f,coarse_u,
                  coarse_extra_in,coarse_extra_out,pre,mid,post);
//            fprintf(stderr,"Coarse solve finished\n");
         } else {
//            fprintf(stderr,"No coarse solve\n");
         }
         MPI_Bcast(&coarse_extra_out,1,MPI_DOUBLE,0,my_comm);
//         if (master()) fprintf(stderr,"V[%d] local coarse correction\n",size_x);
//         SYNC(cerr << coarse_u);
         rebalance_array(coarse_u,local_coarse_2d,my_comm);
         interpolate_grid(correction);
//         cout << "CG: New correction: " << correction;
         apply_operator(correction,f);
         defect = defect-f-coarse_extra_out;
         u = u+correction;
         extra_out = coarse_extra_out;
         break;
      case CYCLE_F:
         coarsen_grid(defect);
         rebalance_array(local_coarse_2d,coarse_f,my_comm);
         if (coarse_solver) {
//            fprintf(stderr,"Coarse solve\n");
            coarse_solver->_cycle(CYCLE_F,coarse_f,coarse_u,
                  extra_in,coarse_extra_out,pre,mid,post);
//            fprintf(stderr,"Coarse solve finished\n");
         } else {
//            fprintf(stderr,"No coarse solve\n");
         }
         MPI_Bcast(&coarse_extra_out,1,MPI_DOUBLE,0,my_comm);
         rebalance_array(coarse_u,local_coarse_2d,my_comm);
         interpolate_grid(correction);
//         cout << "CG: New correction: " << correction;
         apply_operator(correction,f);
//         SYNC(cerr << f; cerr.flush()); MPI_Finalize(); exit(1);
         defect = defect-f-coarse_extra_out;
         u = u+correction;
//         fprintf(stderr,"F(1)[%d] average: %g vs %g\n",size_x,pvsum(u,my_comm)/(size_x*size_z),extra_in);
         extra_out = coarse_extra_out;
         /* Now, two more relaxation steps */
         for (int i = 0; i < mid; i++) {
            do_redblack(defect,correction);
      //      cout << "New correction: " << correction;
            apply_operator(correction,f);
            defect = defect-f;
            u = u+correction;
         }
//         if (1||indefinite_problem) {
//            /* extra_in will change, since part of the mean will already be accounted for */
//            double avg = pvsum(u,my_comm)/(size_x*size_z);
//      //      MPI_Allreduce(&avg,&avg,1,MPI_DOUBLE,MPI_SUM,my_comm);
//            coarse_extra_in = extra_in - avg;
//         }
         coarsen_grid(defect);
         rebalance_array(local_coarse_2d,coarse_f,my_comm);
         if (coarse_solver) {
            coarse_solver->_cycle(CYCLE_V,coarse_f,coarse_u,
                  0,coarse_extra_out,pre,mid,post);
         } else {
         }
         MPI_Bcast(&coarse_extra_out,1,MPI_DOUBLE,0,my_comm);
         rebalance_array(coarse_u,local_coarse_2d,my_comm);
         interpolate_grid(correction);
         apply_operator(correction,f);
         defect = defect-f-coarse_extra_out;
         u = u+correction;
//         fprintf(stderr,"F(2)[%d] average: %g vs %g\n",size_x,pvsum(u,my_comm)/(size_x*size_z),extra_in);
         extra_out += coarse_extra_out;
         break;

      case CYCLE_NONE:
         break;
      default:
         fprintf(stderr,"Invalid cycle type %d\n",cycle);
         abort();
         break;
   }

   /* Now, two more relaxation steps */
   for (int i = 0; i < post; i++) {
      do_redblack(defect,correction);
//      cout << "New correction: " << correction;
      apply_operator(correction,f);
      defect = defect-f;
      u = u+correction;
   }
 //  SYNC(cerr << myrank << ": " << u(u.ubound(firstDim),0) << endl;cerr.flush();)

   // Et voila, return
   return;
}

void MG_Solver::build_sparse_operator() {
   // Build the sparse operator, transposed.  The transpose is useful
   // because it means we can deal with grid points one at a time, rather
   // than having to deal with "influences by a grid point" one at a time.
#define cell_index(a,b) (((a)*size_z)+(b)) 

   assert(nproc == 1);
//   fprintf(stderr,"Sparse building: (%dx%d) %d %d %d\n",size_x,size_z,int(any_dxz),int(bc_tangent),int(bc_normal));
//   double now = MPI_Wtime();

//   cerr << uxx << uzz;
//   cerr << uxz;
//   MPI_Finalize(); exit(1);
   {
      /* Free the symbolic/numeric factors as necessary */
      if (numeric_factor) {
         umfpack_di_free_numeric(&numeric_factor);
         numeric_factor = 0;
      }
      if (symbolic_factor && !coarse_symbolic_ok) {
         umfpack_di_free_symbolic(&symbolic_factor);
         symbolic_factor = 0;
      }
   }

   int entry_count=0; // Count of which matrix entry we're on -- +1 per nonzero

   // If this is an indefinite problem, we add a border of 1s to the matrix on the
   // right and bottom -- this enforces the cell-wise mean constraint and allows
   // for the right-hand side to not satisfy a compatibility condition:
   // u_xx = f (ux(0) = ux(1) = 0) is only defined if int(f,0,1) = 0 normally
   // so we add a phony DoF to make the problem:
   // u_xx = f-<extra>, int(u,0,1) = <defined>, (-ux(0) = ux(1) = -<extra>)
   // This constant is the row/column-number of this extra phantom DoF
   const int norm_cell = cell_index(size_x-1,size_z-1)+1;

   // At this point, everything we care about is processor-local, so
   // the blitz arrays are no longer split.

   /* The left and right boundaries are a special-case, since this code supports
      both periodic, even, odd, and no symmetries in x.  For the general cases in
      the middle of the domain, see the slightly simpler for-loop below this
      block. */
   /* Switch over symmetry type */
   switch (symmetry_type) {
      case SYM_NONE: 
         /* No symmetry, so we're dealing exclusively with boundary conditions */
         {
            int i = 0;
            int j = 0;
            // The bottom left and top left are actually considered to belong
            // to the bottom/top boundaries, not the left-right
            A_cols[cell_index(i,j)] = entry_count;
            // bottom-left
            A_double[entry_count] = ux_bot(i)*Dx(i,1)+uz_bot(i)*Dz(j,1)+u_bot(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            // up-from-bottom - left
            if (bc_normal) {
               A_double[entry_count] = uz_bot(i)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
            }
            if (bc_tangent) {
               // right-bottom
               A_double[entry_count] = ux_bot(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }

            // Now, loop over the interior of the left boundary
            for (j = 1; j < size_z - 1; j++) {
               A_cols[cell_index(i,j)] = entry_count;
               if (bc_tangent) {
                  // below
                  A_double[entry_count] = uz_left(j)*Dz(j,0);
                  A_rows[entry_count] = cell_index(i,j-1);
                  entry_count++;
               }
               // middle
               A_double[entry_count] = uz_left(j)*Dz(j,1)+ux_left(j)*Dx(i,1)+u_left(j);
               A_rows[entry_count] = cell_index(i,j);
               entry_count++;
               if (bc_tangent) {
                  // upper
                  A_double[entry_count] = uz_left(j)*Dz(j,2);
                  A_rows[entry_count] = cell_index(i,j+1);
                  entry_count++;
               }
               if (bc_normal) {
                  // right
                  A_double[entry_count] = ux_left(j)*Dx(i,2);
                  A_rows[entry_count] = cell_index(i+1,j);
                  entry_count++;
               }
               if (indefinite_problem) {
                  A_double[entry_count] = 1;
                  A_rows[entry_count] = norm_cell;
                  entry_count++;
               }
            }
            // Now, for the top left 
            j = size_z-1;
            // The bottom left and top left are actually considered to belong
            // to the bottom/top boundaries, not the left-right
            A_cols[cell_index(i,j)] = entry_count;
            // down-from-top - left
            if (bc_normal) {
               A_double[entry_count] = uz_top(i)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
            }
            // top-left
            A_double[entry_count] = ux_top(i)*Dx(i,1)+uz_top(i)*Dz(j,1)+u_top(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            if (bc_tangent) {
               // right-top
               A_double[entry_count] = ux_top(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
         } break;
      case SYM_EVEN:
      case SYM_ODD:
         /* Even and odd symmetry work in the same way -- the interior of the left
            boundary works much like the interior of the grid, save that the point
            to the right is the same as (SYM_EVEN) or the negative of (SYM_ODD)
            the "phantom" point on the left. */
         {
            int sym = (symmetry_type == SYM_EVEN ? 1 : -1);
            int i = 0;
            int j = 0;
            A_cols[cell_index(i,j)] = entry_count;
            // bottom-left
            A_double[entry_count] = ux_bot(i)*(Dx(i,1)+sym*Dx(i,0))+uz_bot(i)*Dz(j,1)+u_bot(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            // up-from-bottom - left
            if (bc_normal) {
               A_double[entry_count] = uz_bot(i)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
            }
            if (bc_tangent) {
               // right-bottom
               A_double[entry_count] = ux_bot(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
            for (j = 1; j < size_z-1; j++) { // middle
               A_cols[cell_index(i,j)] = entry_count;
               // middle-lower
               A_double[entry_count] = uxz(i,j)*(Dx(i,1)+sym*Dx(i,0))*Dz(j,0)+uzz(i,j)*Dzz(j,0)+uz(i,j)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
               // middle-middle
               A_double[entry_count] = uxz(i,j)*(Dx(i,1)+sym*Dx(i,0))*Dz(j,1) + 
                  uxx(i,j)*(Dxx(i,1)+sym*Dxx(i,0)) + uzz(i,j)*Dzz(j,1) +
                  ux(i,j)*(Dx(i,1)+sym*Dx(i,0)) + uz(i,j)*Dz(j,1) + helm_parameter;
               A_rows[entry_count] = cell_index(i,j);
               entry_count++;
               // middle-upper
               A_double[entry_count] = uxz(i,j)*(Dx(i,1)+sym*Dx(i,0))*Dz(j,2)+uzz(i,j)*Dzz(j,2)+uz(i,j)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
               if (any_dxz) {
                  // right-lower
                  A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,0);
                  A_rows[entry_count] = cell_index(i+1,j-1);
                  entry_count++;
               }
               // right-middle
               A_double[entry_count] = uxz(i,j)*(Dx(i,2))*Dz(j,1)+
                                       uxx(i,j)*(Dxx(i,2))+
                                       ux(i,j)*(Dx(i,2));
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
               if (any_dxz) {
                  // right-upper
                  A_double[entry_count] = uxz(i,j)*(Dx(i,2))*Dz(j,2);
                  A_rows[entry_count] = cell_index(i+1,j+1);
                  entry_count++;
               }
               if (indefinite_problem) {
                  A_double[entry_count] = 1;
                  A_rows[entry_count] = norm_cell;
                  entry_count++;
               }
            }
            j = size_z-1; // Top left point
            A_cols[cell_index(i,j)] = entry_count;
            // down-from-top - left
            if (bc_normal) {
               A_double[entry_count] = uz_top(i)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
            }
            // top-left
            A_double[entry_count] = ux_top(i)*(Dx(i,1)+sym*Dx(i,0))+uz_top(i)*Dz(j,1)+u_top(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            if (bc_tangent) {
               // right-top
               A_double[entry_count] = ux_top(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
         }
         break;
      case SYM_PERIODIC:
         {
            /* The periodic case works just like the interior (see below), but
               with the caveat that the "left" points are really located at
               i = size_x-1.  This affects the order we enter them into the
               matrix, since UMFPACK demands that the entires be in lexical order. */
            int i = 0;
            int j = 0; // bottom
            // Handle bottom BCs
            A_cols[cell_index(i,j)] = entry_count;
            // bottom middle
            A_double[entry_count] = ux_bot(i)*Dx(i,1)+uz_bot(i)*Dz(j,1)+u_bot(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            if (bc_normal) {
               // up-from-bottom middle
               A_double[entry_count] = uz_bot(i)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
            }
            if (bc_tangent) {
               // bottom right
               A_double[entry_count] = ux_bot(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
            }
            if (bc_tangent) {
               // bottom left
               A_double[entry_count] = ux_bot(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(size_x-1,j);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }

            for (j = 1; j < size_z-1; j++) { // middle
               A_cols[cell_index(i,j)] = entry_count;
               // middle-lower
               A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,0)+uzz(i,j)*Dzz(j,0)+uz(i,j)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
               // middle-middle
               A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,1) + uxx(i,j)*Dxx(i,1) + uzz(i,j)*Dzz(j,1) +
                  ux(i,j)*Dx(i,1) + uz(i,j)*Dz(j,1) + helm_parameter;
               A_rows[entry_count] = cell_index(i,j);
               entry_count++;
               // middle-upper
               A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,2)+uzz(i,j)*Dzz(j,2)+uz(i,j)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
               if (any_dxz) {
                  // right-lower
                  A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,0);
                  A_rows[entry_count] = cell_index(i+1,j-1);
                  entry_count++;
               }
               // right-middle
               A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,1)+uxx(i,j)*Dxx(i,2)+ux(i,j)*Dx(i,2);
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
               if (any_dxz) {
                  // right-upper
                  A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,2);
                  A_rows[entry_count] = cell_index(i+1,j+1);
                  entry_count++;
               }
               if (any_dxz) {
                  // left-lower
                  A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,0);
                  A_rows[entry_count] = cell_index(size_x-1,j-1);
                  entry_count++;
               }
               // left-middle
               A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,1)+uxx(i,j)*Dxx(i,0)+ux(i,j)*Dx(i,0);
               A_rows[entry_count] = cell_index(size_x-1,j);
               entry_count++;
               if (any_dxz) {
                  // left-upper
                  A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,2);
                  A_rows[entry_count] = cell_index(size_x-1,j+1);
                  entry_count++;
               }
               if (indefinite_problem) {
                  A_double[entry_count] = 1;
                  A_rows[entry_count] = norm_cell;
                  entry_count++;
               }
            }
            j = size_z-1; // top
            // Handle top BCs
            A_cols[cell_index(i,j)] = entry_count;
            if (bc_normal) {
               // down-from-top middle
               A_double[entry_count] = uz_top(i)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
            }
            // top middle
            A_double[entry_count] = ux_top(i)*Dx(i,1)+uz_top(i)*Dz(j,1)+u_top(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            // top right
            if (bc_tangent) {
               A_double[entry_count] = ux_top(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(i+1,j);
               entry_count++;
            }
            if (bc_tangent) {
               // top left
               A_double[entry_count] = ux_top(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(size_x-1,j);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
         } break;
   }

   //-----------------------------------------
   //The interior is much simpler, as there
   //are no symmetry-special-cases to worry
   //about.  This ultimately implements a
   //kind of Sparse Kroneker Product, though
   //from a grid-centric rather than vector-
   //centric point of view.
   //----------------------------------------

   for (int i = 1; i < size_x-1; i++) { // Loop over each cell in its "lexical" order
      int j = 0; // bottom
      // Handle bottom BCs
      A_cols[cell_index(i,j)] = entry_count;
      if (bc_tangent) {
         // bottom left
         A_double[entry_count] = ux_bot(i)*Dx(i,0);
         A_rows[entry_count] = cell_index(i-1,j);
         entry_count++;
      }
      // bottom middle
      A_double[entry_count] = ux_bot(i)*Dx(i,1)+uz_bot(i)*Dz(j,1)+u_bot(i);
      A_rows[entry_count] = cell_index(i,j);
      entry_count++;
      if (bc_normal) {
         // up-from-bottom middle
         A_double[entry_count] = uz_bot(i)*Dz(j,2);
         A_rows[entry_count] = cell_index(i,j+1);
         entry_count++;
      }
      if (bc_tangent) {
         // bottom right
         A_double[entry_count] = ux_bot(i)*Dx(i,2);
         A_rows[entry_count] = cell_index(i+1,j);
         entry_count++;
      }
      if (indefinite_problem) {
         A_double[entry_count] = 1;
         A_rows[entry_count] = norm_cell;
         entry_count++;
      }
      
      for (j = 1; j < size_z-1; j++) { // middle
         A_cols[cell_index(i,j)] = entry_count;
         if (any_dxz) {
            // left-lower
            A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,0);
            A_rows[entry_count] = cell_index(i-1,j-1);
            entry_count++;
         }
         // left-middle
         A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,1)+uxx(i,j)*Dxx(i,0)+ux(i,j)*Dx(i,0);
         A_rows[entry_count] = cell_index(i-1,j);
         entry_count++;
         if (any_dxz) {
            // left-upper
            A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,2);
            A_rows[entry_count] = cell_index(i-1,j+1);
            entry_count++;
         }
         // middle-lower
         A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,0)+uzz(i,j)*Dzz(j,0)+uz(i,j)*Dz(j,0);
         A_rows[entry_count] = cell_index(i,j-1);
         entry_count++;
         // middle-middle
//         fprintf(stderr,"(%d,%d) mm (%.2f [%.2f] %.2f [%.2f] %.2f [%.2f]) + %.2f = ",i,j,uxz(i,j),Dx(i,1)*Dz(j,1),uxx(i,j),Dxx(i,1),uzz(i,j),Dzz(j,1),helm_parameter);
         A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,1) + uxx(i,j)*Dxx(i,1) + uzz(i,j)*Dzz(j,1) +
                                 ux(i,j)*Dx(i,1) + uz(i,j)*Dz(j,1) + helm_parameter;
         A_rows[entry_count] = cell_index(i,j);
//         fprintf(stderr,"%f\n",A_double[entry_count]);
         entry_count++;
         // middle-upper
         A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,2)+uzz(i,j)*Dzz(j,2)+uz(i,j)*Dz(j,2);
         A_rows[entry_count] = cell_index(i,j+1);
         entry_count++;
         if (any_dxz) {
            // right-lower
            A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,0);
            A_rows[entry_count] = cell_index(i+1,j-1);
            entry_count++;
         }
         // right-middle
         A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,1)+uxx(i,j)*Dxx(i,2)+ux(i,j)*Dx(i,2);
         A_rows[entry_count] = cell_index(i+1,j);
         entry_count++;
         if (any_dxz) {
            // right-upper
            A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,2);
            A_rows[entry_count] = cell_index(i+1,j+1);
            entry_count++;
         }
         if (indefinite_problem) {
            A_double[entry_count] = 1;
            A_rows[entry_count] = norm_cell;
            entry_count++;
         }
      }
      j = size_z-1; // top
      // Handle top BCs
      A_cols[cell_index(i,j)] = entry_count;
      if (bc_tangent) {
         // top left
         A_double[entry_count] = ux_top(i)*Dx(i,0);
         A_rows[entry_count] = cell_index(i-1,j);
         entry_count++;
      }
      if (bc_normal) {
         // down-from-top middle
         A_double[entry_count] = uz_top(i)*Dz(j,0);
         A_rows[entry_count] = cell_index(i,j-1);
         entry_count++;
      }
      // top middle
      A_double[entry_count] = ux_top(i)*Dx(i,1)+uz_top(i)*Dz(j,1)+u_top(i);
      A_rows[entry_count] = cell_index(i,j);
      entry_count++;
      // top right
      if (bc_tangent) {
         A_double[entry_count] = ux_top(i)*Dx(i,2);
         A_rows[entry_count] = cell_index(i+1,j);
         entry_count++;
      }
      if (indefinite_problem) {
         A_double[entry_count] = 1;
         A_rows[entry_count] = norm_cell;
         entry_count++;
      }
    
      
   }

   /*
      -----------------------------------------------
      Finally, handle the right boundary.  Just as with
      the left boundary, it requires switching based
      on the problem's symmetry.
      ------------------------------------------------
      */
   /* Switch over symmetry type */
   switch (symmetry_type) {
      case SYM_NONE: 
         /* No symmetry, so we're dealing exclusively with boundary conditions */
         {
            int i = size_x-1;
            int j = 0;
            // The bottom right and top right are actually considered to belong
            // to the bottom/top boundaries, not the left-right
            A_cols[cell_index(i,j)] = entry_count;
            // --- handle the bottom right corner ---
            if (bc_tangent) {
               // left-bottom
               A_double[entry_count] = ux_bot(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
            }
            // middle
            A_double[entry_count] = ux_bot(i)*Dx(i,1)+uz_bot(i)*Dz(j,1)+u_bot(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            // up-from-bottom-right
            if (bc_normal) {
               A_double[entry_count] = uz_bot(i)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }

            // Now, loop over the interior of the right boundary
            for (j = 1; j < size_z - 1; j++) {
               A_cols[cell_index(i,j)] = entry_count;
               if (bc_normal) {
                  // left
                  A_double[entry_count] = ux_right(j)*Dx(i,0);
                  A_rows[entry_count] = cell_index(i-1,j);
                  entry_count++;
               }
               if (bc_tangent) {
                  // below
                  A_double[entry_count] = uz_right(j)*Dz(j,0);
                  A_rows[entry_count] = cell_index(i,j-1);
                  entry_count++;
               }
               // middle
               A_double[entry_count] = uz_right(j)*Dz(j,1)+ux_right(j)*Dx(i,1)+u_right(j);
               A_rows[entry_count] = cell_index(i,j);
               entry_count++;
               if (bc_tangent) {
                  // upper
                  A_double[entry_count] = uz_right(j)*Dz(j,2);
                  A_rows[entry_count] = cell_index(i,j+1);
                  entry_count++;
               }
               if (indefinite_problem) {
                  A_double[entry_count] = 1;
                  A_rows[entry_count] = norm_cell;
                  entry_count++;
               }
            }
            // Now, for the top right
            j = size_z-1;
            // The bottom right and top right are actually considered to belong
            // to the bottom/top boundaries, not the left-right
            A_cols[cell_index(i,j)] = entry_count;
            if (bc_tangent) {
               // left-top
               A_double[entry_count] = ux_top(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
            }
            // down-from-top-right
            if (bc_normal) {
               A_double[entry_count] = uz_top(i)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
            }
            // top-right
            A_double[entry_count] = ux_top(i)*Dx(i,1)+uz_top(i)*Dz(j,1)+u_top(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
      
         } 
         
         
         break;
      case SYM_EVEN:
      case SYM_ODD:
         /* Even and odd symmetry work in the same way -- the interior of the left
            boundary works much like the interior of the grid, save that the point
            to the right is the same as (SYM_EVEN) or the negative of (SYM_ODD)
            the "phantom" point on the left. */
         {
            int sym = (symmetry_type == SYM_EVEN ? 1 : -1);
            int i = size_x-1;
            int j = 0;
            A_cols[cell_index(i,j)] = entry_count;
            if (bc_tangent) {
               // left-bottom
               A_double[entry_count] = ux_bot(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
            }
            // bottom-right
            A_double[entry_count] = ux_bot(i)*(Dx(i,1)+sym*Dx(i,2))+uz_bot(i)*Dz(j,1)+u_bot(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            // up-from-bottom-right
            if (bc_normal) {
               A_double[entry_count] = uz_bot(i)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
            for (j = 1; j < size_z-1; j++) { // middle
               A_cols[cell_index(i,j)] = entry_count;
               if (any_dxz) {
                  // left-lower
                  A_double[entry_count] = uxz(i,j)*(Dx(i,0))*Dz(j,0);
                  A_rows[entry_count] = cell_index(i-1,j-1);
                  entry_count++;
               }
               // left-middle
               A_double[entry_count] = uxz(i,j)*(Dx(i,0))*Dz(j,1)+
                                       uxx(i,j)*(Dxx(i,0))+
                                       ux(i,j)*(Dx(i,0));
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
               if (any_dxz) {
                  // left-upper
                  A_double[entry_count] = uxz(i,j)*(Dx(i,0))*Dz(j,2);
                  A_rows[entry_count] = cell_index(i-1,j+1);
                  entry_count++;
               }
               // middle-lower
               A_double[entry_count] = uxz(i,j)*(Dx(i,1)+sym*Dx(i,2))*Dz(j,0)+uzz(i,j)*Dzz(j,0)+uz(i,j)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
               // middle-middle
               A_double[entry_count] = uxz(i,j)*(Dx(i,1)+sym*Dx(i,2))*Dz(j,1) + 
                     uxx(i,j)*(Dxx(i,1)+sym*Dxx(i,2)) + uzz(i,j)*Dzz(j,1) +
                  ux(i,j)*(Dx(i,1)+sym*Dxx(i,2)) + uz(i,j)*Dz(j,1) + helm_parameter;
               A_rows[entry_count] = cell_index(i,j);
               entry_count++;
               // middle-upper
               A_double[entry_count] = uxz(i,j)*(Dx(i,1)+sym*Dx(i,2))*Dz(j,2)+uzz(i,j)*Dzz(j,2)+uz(i,j)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
               if (indefinite_problem) {
                  A_double[entry_count] = 1;
                  A_rows[entry_count] = norm_cell;
                  entry_count++;
               }
            }
            j = size_z-1; // Top right point
            A_cols[cell_index(i,j)] = entry_count;
            if (bc_tangent) {
               // left-top
               A_double[entry_count] = ux_top(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
            }
            // down-from-top-right
            if (bc_normal) {
               A_double[entry_count] = uz_top(i)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
            }
            // top-right
            A_double[entry_count] = ux_top(i)*(Dx(i,1)+sym*Dx(i,2))+uz_top(i)*Dz(j,1)+u_top(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
         }
         break;
      case SYM_PERIODIC:
         {
            /* The periodic case works just like the interior (see above), but
               with the caveat that the "right" points are really located at
               i = 0.  This affects the order we enter them into the
               matrix, since UMFPACK demands that the entires be in lexical order. */
            int i = size_x-1;
            int j = 0; // bottom
            // Handle bottom BCs
            A_cols[cell_index(i,j)] = entry_count;
            if (bc_tangent) {
               // bottom right
               A_double[entry_count] = ux_bot(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(0,j);
               entry_count++;
            }
            if (bc_tangent) {
               // bottom left
               A_double[entry_count] = ux_bot(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
            }
            // bottom middle
            A_double[entry_count] = ux_bot(i)*Dx(i,1)+uz_bot(i)*Dz(j,1)+u_bot(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            if (bc_normal) {
               // up-from-bottom middle
               A_double[entry_count] = uz_bot(i)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
            }
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }

            for (j = 1; j < size_z-1; j++) { // middle
               A_cols[cell_index(i,j)] = entry_count;
               if (any_dxz) {
                  // right-lower
                  A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,0);
                  A_rows[entry_count] = cell_index(0,j-1);
                  entry_count++;
               }
               // right-middle
               A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,1)+uxx(i,j)*Dxx(i,2)+ux(i,j)*Dx(i,2);
               A_rows[entry_count] = cell_index(0,j);
               entry_count++;
               if (any_dxz) {
                  // right-upper
                  A_double[entry_count] = uxz(i,j)*Dx(i,2)*Dz(j,2);
                  A_rows[entry_count] = cell_index(0,j+1);
                  entry_count++;
               }
               if (any_dxz) {
                  // left-lower
                  A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,0);
                  A_rows[entry_count] = cell_index(i-1,j-1);
                  entry_count++;
               }
               // left-middle
               A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,1)+uxx(i,j)*Dxx(i,0)+ux(i,j)*Dx(i,0);
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
               if (any_dxz) {
                  // left-upper
                  A_double[entry_count] = uxz(i,j)*Dx(i,0)*Dz(j,2);
                  A_rows[entry_count] = cell_index(i-1,j+1);
                  entry_count++;
               }
               // middle-lower
               A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,0)+uzz(i,j)*Dzz(j,0)+uz(i,j)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
               // middle-middle
               A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,1) + uxx(i,j)*Dxx(i,1) + uzz(i,j)*Dzz(j,1) +
                  ux(i,j)*Dx(i,1) + uz(i,j)*Dz(j,1) + helm_parameter;
               A_rows[entry_count] = cell_index(i,j);
               entry_count++;
               // middle-upper
               A_double[entry_count] = uxz(i,j)*Dx(i,1)*Dz(j,2)+uzz(i,j)*Dzz(j,2)+uz(i,j)*Dz(j,2);
               A_rows[entry_count] = cell_index(i,j+1);
               entry_count++;
               if (indefinite_problem) {
                  A_double[entry_count] = 1;
                  A_rows[entry_count] = norm_cell;
                  entry_count++;
               }
            }
            j = size_z-1; // top
            // Handle top BCs
            A_cols[cell_index(i,j)] = entry_count;
            // top right
            if (bc_tangent) {
               A_double[entry_count] = ux_top(i)*Dx(i,2);
               A_rows[entry_count] = cell_index(0,j);
               entry_count++;
            }
            if (bc_tangent) {
               // top left
               A_double[entry_count] = ux_top(i)*Dx(i,0);
               A_rows[entry_count] = cell_index(i-1,j);
               entry_count++;
            }
            if (bc_normal) {
               // down-from-top middle
               A_double[entry_count] = uz_top(i)*Dz(j,0);
               A_rows[entry_count] = cell_index(i,j-1);
               entry_count++;
            }
            // top middle
            A_double[entry_count] = ux_top(i)*Dx(i,1)+uz_top(i)*Dz(j,1)+u_top(i);
            A_rows[entry_count] = cell_index(i,j);
            entry_count++;
            if (indefinite_problem) {
               A_double[entry_count] = 1;
               A_rows[entry_count] = norm_cell;
               entry_count++;
            }
         } break;
   }
   if (indefinite_problem) {
      /* Finally, if this is an indefinite problem then we need to add the
         other half of the corresponding border, to ensure that the pointwise
         mean is set properly */
      A_cols[norm_cell] = entry_count;
      for (int i = 0; i < norm_cell; i++) {
         A_double[entry_count] = 1.0/(size_x*size_z);
         A_rows[entry_count] = i;
         entry_count++;
      }
   A_cols[norm_cell+1] = entry_count;
   } else {
      A_cols[norm_cell] = entry_count;
   }
   sparse_size = entry_count;

   /* Now, factor the operator for solving */
   {
      int status, n, m;
      n = size_x*size_z + (indefinite_problem ? 1 : 0);
      m = n;
      double Control[UMFPACK_CONTROL], Info[UMFPACK_INFO];
//      for (int i = 0; i < size_x; i++) {
//         for (int j = 0; j < size_z; j++) {
//            fprintf(stderr,"(%f %f) ",ux(i,j),uz(i,j));
//         } fprintf(stderr,"\n");
//      }
//      for (int i = 0; i < size_x; i++) {
//         for (int j = 0; j < size_z; j++) {
//            fprintf(stderr,"(%f %f) ",uxx(i,j),uzz(i,j));
//         } fprintf(stderr,"\n");
//      }
//      for (int i = 0; i < size_x; i++) {
//         fprintf(stderr,"[%f %f %f]\n",Dxx(i,0),Dxx(i,1),Dxx(i,2));
//      }
//      for (int i = 0; i < size_z; i++) {
//         fprintf(stderr,"[%f %f %f]\n",Dzz(i,0),Dzz(i,1),Dzz(i,2));
//      }
//      fprintf(stderr,"[%d,%d] EOM\n",n,A_cols[n]);
      umfpack_di_defaults(Control);
      Control[UMFPACK_DROPTOL] = 1e-16;
//      Control[UMFPACK_PRL] = 3;
//      Control[UMFPACK_SYM_PIVOT_TOLERANCE] = 1;
//      Control[UMFPACK_PIVOT_TOLERANCE] = 1;
      if (!symbolic_factor || !coarse_symbolic_ok) {
         status = umfpack_di_symbolic(m,n,A_cols,A_rows,A_double,&symbolic_factor,Control,Info);
//         fprintf(stderr,"Symbolic factor (%d) status: %d in %.2f sec\n",status,Info[UMFPACK_STRATEGY_USED],Info[UMFPACK_SYMBOLIC_TIME]);
         if (status) {
            fprintf(stderr,"Nonzero status (%d) on symbolic factorization!\n",status);
            FILE* errf = fopen("matrix_dump.txt","w");
            for (int i = 0; i < n; i++) {
               for (int j = A_cols[i]; j< A_cols[i+1]; j++) {
                  fprintf(errf,"%d %d %.15e\n",i,A_rows[j],A_double[j]);
                  fprintf(stderr,"(%d,%d): %.2g ",i,A_rows[j],A_double[j]);
                  if (!((j-A_cols[i]+1)%16)) fprintf(stderr,"\n     ");
               }
               fprintf(stderr,"\n");
            }
            fclose(errf);
         }
         assert (status == 0);
      }
      status = umfpack_di_numeric(A_cols,A_rows,A_double,symbolic_factor,&numeric_factor,Control,Info);
//      fprintf(stderr,"Numeric factor status: %d in %.2f sec\n",status,Info[UMFPACK_NUMERIC_TIME]);
      if (status) {
         fprintf(stderr,"Nonzero status (%d) on numeric factorization!\n",status);
         FILE* errf = fopen("matrix_dump.txt","w");
         for (int i = 0; i < n; i++) {
            for (int j = A_cols[i]; j< A_cols[i+1]; j++) {
               fprintf(errf,"%d %d %.15e\n",i,A_rows[j],A_double[j]);
               fprintf(stderr,"(%d,%d): %.2g ",i,A_rows[j],A_double[j]);
               if (!((j-A_cols[i]+1)%16)) fprintf(stderr,"\n     ");
            }
            fprintf(stderr,"\n");
         }
         fclose(errf);
      }
//      fprintf(stderr,"Numreic factor nonzeros: %.0f (%.0f dropped)\n",Info[UMFPACK_LNZ]+Info[UMFPACK_UNZ],Info[UMFPACK_NZDROPPED]);
//      fprintf(stderr,"Numeric factor fill-in: %.2f\n",(Info[UMFPACK_LNZ]+Info[UMFPACK_UNZ])/(n));
      assert (status == 0);
      coarse_symbolic_ok = true;
      coarse_numeric_ok = true;
   }
//   double later = MPI_Wtime();
//   fprintf(stderr,"Coarse operator construction took %g sec\n",later-now);

}

void MG_Solver::check_bc_consistency() {
   /* Defined on only the coarsest level, checks to see wheteher the BCs are consistent
      with the booleans that govern the array stencil & whether the coarse operator
      needs re-symbolic-factoring */
   // tangent bdy derivative
//   fprintf(stderr,"BC consistency check\n");
   if (any(ux_top) || any(ux_bot) || (symmetry_type == SYM_NONE && (any(uz_left) || any(uz_right)))) {
      if (!bc_tangent) coarse_symbolic_ok = false;
      bc_tangent = true;
   } else {
      if (bc_tangent) coarse_symbolic_ok = false;
      bc_tangent = false;
   }
   // normal bdy derivative
   if (any(uz_top) || any(uz_bot) || (symmetry_type == SYM_NONE && (any(ux_left) || any(ux_right)))) {
      if (!bc_normal) coarse_symbolic_ok = false;
      bc_normal = true;
   } else {
      if (bc_normal) coarse_symbolic_ok = false;
      bc_normal = false;
   }
   // indefinite problem
   if (!any(u_top) && !any(u_bot) && 
            ((symmetry_type == SYM_NONE && !any(u_left) && !any(u_right)) ||
             (symmetry_type == SYM_EVEN) ||
             (symmetry_type == SYM_PERIODIC)) &&
            helm_parameter == 0) {
      if (!indefinite_problem) coarse_symbolic_ok = false;
//      fprintf(stderr,"Setting indefinite problem\n");
      indefinite_problem = true;
   } else {
      if (indefinite_problem) coarse_symbolic_ok = false;
//      fprintf(stderr,"Setting NOT indefinite problem\n");
      indefinite_problem = false;
   }


}
