#include "TArray.hpp"
#include "blitz/array.h"
#include "blitz/tinyvec-et.h"
#include "ESolver.hpp"
#include "gmres_1d_solver.hpp"
#include "gmres_2d_solver.hpp"
#include "Par_util.hpp"
#include <iostream>
#include "stdio.h"
#include "grad.hpp"

using namespace std;
using namespace Transformer;
//extern int plan_counts;
namespace ESolver {
//   using namespace TArray;
   using TArrayn::Grad;
   using blitz::Array;
   using blitz::cast;
   
   blitz::firstIndex ii;
   blitz::secondIndex jj;
   blitz::thirdIndex kk;

   ElipSolver::ElipSolver(double M, TransWrapper * spec,Grad * in_grad)
      : M(M), spec_transform(spec), gradient(in_grad),
      cTransposer(0), rTransposer(0), rtransyz(0), ctransyz(0),
      twod_u(0), twod_f(0)
   {
      
      /* The actual spectral work here is farmed out to the Transformer class.
         Wavenumbers will be generated when used by solve. */
   }
   ElipSolver::~ElipSolver() { // Empty destructor
      if (cTransposer) delete cTransposer;
      if (rTransposer) delete rTransposer;
      if (rtransyz) delete rtransyz;
      if (ctransyz) delete ctransyz;
      if (twod_u) delete twod_u;
      if (twod_f) delete twod_f;
   }

   void ElipSolver::solve(DTArray & rhs, DTArray & output,
         S_EXP type_x, S_EXP type_y, S_EXP type_z,
         double zbc_a, double zbc_b, double xbc_a, double xbc_b,
         double ybc_a, double ybc_b) {
      if (xbc_a == 0 && xbc_b == 0) {
         xbc_a = zbc_a; xbc_b = zbc_b;
      } if (ybc_a == 0 && ybc_b == 0) {
         ybc_a = zbc_a; ybc_b = zbc_b;
      }

//      int plan_before = plan_counts;
      // Select which solver we're using
      if (type_z == CHEBY && type_x != CHEBY && type_y != CHEBY &&
            gradient->constant_diagonal_jac())
            {
         /* With no coordinate mapping, a Chebyshev expansion in the vertical, and
            a trig expansion of some sort in the horizontal, use the specialized
            1D solver (FD preconditioner, GMRES) */
         chebz_solve(rhs, output, type_x, type_y, zbc_a, zbc_b);
         return;
      } else if (gradient->constant_diagonal_jac() && type_x != CHEBY && type_z != CHEBY){
         /* With a triple-trig expansion, the solve is algebraic under the proper
            transform.  Boundary conditions are also implied by the symmetry, so
            the given ones are unused. */
         threespec_solve(rhs, output, type_x, type_y, type_z);
         return;
      } else {
         /* Otherwise, use the more general 2D-multigrid solver */
         if (type_z != CHEBY) {
            if (master()) 
               fprintf(stderr,"ERROR: Multigrid/GMRES solver only supported with Chebyshev vertical\n");
            abort();
         }
         if (type_y == CHEBY || !gradient->constant_jac(secondDim,firstDim) ||
               !gradient->constant_jac(secondDim,secondDim) ||
               !gradient->constant_jac(secondDim,thirdDim)) {
            if (master())
               fprintf(stderr,"ERROR: Complicated spanwise mappings are currently unsupported\n");
            abort();
         }
         twodeemg_solve(rhs,output,type_x,type_y,zbc_a, zbc_b, xbc_a, xbc_b);
         return;

      }
   }


   /* Specialization for the x/z multigrid case, with y being trig/Fourier in 
      some way */
   void ElipSolver::twodeemg_solve(DTArray & rhs, DTArray & output,
         S_EXP type_x, S_EXP type_y, double zbc_a, double zbc_b, double xbc_a, double xbc_b) {
      // Get the spanwise transform via the gradient operator
      Trans1D * spanwise_xform = gradient->get_trans(secondDim);

      int szx, szy, szz;
      if (spanwise_xform) {
         // Apply the spanwise transform
         spanwise_xform->forward_transform(&rhs,type_y);
         spanwise_xform->get_sizes(szx,szy,szz);
      } else {
         // Operating with no spanwise xform, so probably a 2D problem.
         gradient->get_sizes(szx,szy,szz);
      }

      /* Get the constant from the (2,2) entry of the Jacobian */
      double Jy; DTArray * unused;
      gradient->get_jac(secondDim,secondDim,Jy,unused);

      int dim2_extent = szy;
      if (spanwise_xform && spanwise_xform->use_complex()) {
         dim2_extent = spanwise_xform->get_complex_temp()->extent(secondDim);
      }

      Array<double,1> ls(dim2_extent);
      if (spanwise_xform)
         ls = spanwise_xform->wavenums();
      else ls = 0;


      /* As usual, the general problem here is
         Lap * u - helm*u = f. 

         In this specialization, we get to assume that Lap somehow
         couples x and z and they cannot be solved for algebraically.
         However, the spanwise coordinate (y) -is- separable and in
         some way trigonometric, so we can say:

         Lap_xz u + u_yy - helm*u = f

         After the appropriate Fourier transform:

         Lap_xz uhat - (helm + l^2) uhat = fhat

         where l^2 is the spanwise wavenumbers, accounting for the grid
         length (constant Jacobian entry (2,2)) */


      /* Now, if we have a full 3D problem to deal with (more than one
         spanwise point), we still only want to operate on a single 2D
         array at a time.  So, create and allocate a slab of the proper form.
         Small note here that a complex spanwise transform needs this slab
         alloction, even if it has only a single (complex) point -- the
         seperate real/imaginary components don't play nicely with the need
         for DTArrays. */
      if (szy > 1 || (spanwise_xform && spanwise_xform->use_complex())) {
         /* Find base and extent of the 2D arrays, from the transformed RHS */
         blitz::TinyVector<int,3> twod_base, twod_extent;
         blitz::GeneralArrayStorage<3> storage_order;
         if (spanwise_xform->use_complex()) {
            storage_order.ordering() = spanwise_xform->get_complex_temp()->ordering();
            twod_base(0) = spanwise_xform->get_complex_temp()->lbound(firstDim);
            twod_base(2) = spanwise_xform->get_complex_temp()->lbound(thirdDim);
            twod_extent(0) = spanwise_xform->get_complex_temp()->extent(firstDim);
            twod_extent(2) = spanwise_xform->get_complex_temp()->extent(thirdDim);
            twod_base(1) = 0; twod_extent(1) = 0;
         } else { // real temporary
            storage_order.ordering() = spanwise_xform->get_real_temp()->ordering();
            twod_base(0) = spanwise_xform->get_real_temp()->lbound(firstDim);
            twod_base(2) = spanwise_xform->get_real_temp()->lbound(thirdDim);
            twod_extent(0) = spanwise_xform->get_real_temp()->extent(firstDim);
            twod_extent(2) = spanwise_xform->get_real_temp()->extent(thirdDim);
         }
         twod_base(1) = 0; twod_extent(1) = 1;
         
         /* If they don't already exist, allocate the 2D temporaries.  The
            default C-storage order will suffice. */
         if (!twod_u) {
            twod_u = new DTArray (twod_base,twod_extent,storage_order);
         }
         if (!twod_f) {
            twod_f = new DTArray (twod_base, twod_extent,storage_order);
         }
         // Make sure that everything matches

         assert(all(twod_u->lbound() == twod_base) && all(twod_u->extent() == twod_extent));
         assert(all(twod_f->lbound() == twod_base) && all(twod_f->extent() == twod_extent));
      } else if (szy > 1 && !spanwise_xform) {
         if (master()) {
            fprintf(stderr,"Currently no support for 3D problems with a null spanwise transform.\n");
            fprintf(stderr,"Please implement this before trying a multi-layer shallow water model or whatever you're doing.\n");
         } abort();
      } else if (szy == 1) {
         if (output.data() == rhs.data()) {
            // 2D-multigrid doesn't like an in-place transform, so copy the RHS to twod_f, allocating
            // if necessary
            if (!twod_f) {
               twod_f = alloc_array(szx,szy,szz,gradient->get_communicator());
            }
            *twod_f = rhs;
         }
      }
      /* Note that if szy == 1 and the spanwise transform is real, then we have a morally
         2D array already and don't need to allocate any kind of slab */

      /* Now, loop over the spanwise variable */
      int lowj, highj;
      if (!spanwise_xform) {
         lowj = highj = 0;
      } else if (spanwise_xform->use_complex()) {
         lowj = spanwise_xform->get_complex_temp()->lbound(secondDim);
         highj = spanwise_xform->get_complex_temp()->ubound(secondDim);
      } else {
         lowj = spanwise_xform->get_real_temp()->lbound(secondDim);
         highj = spanwise_xform->get_real_temp()->ubound(secondDim);
      }
      for (int j = lowj; j <= highj; j++) {
         /* If y has a real transform, the process is relatively simple: */
         // Array pointers for the solve, to mask whether or not we have to use
         // a slab
         using blitz::Range;
         DTArray * u_ptr, * f_ptr;
         if (!spanwise_xform || !spanwise_xform->use_complex()) {
            if (szy > 1) {
               *twod_f = (*(spanwise_xform->get_real_temp()))
                              (Range::all(),Range(j,j),Range::all());
               f_ptr = twod_f;
               u_ptr = twod_u;
            } else if (spanwise_xform) {
               u_ptr = &output;
               f_ptr = spanwise_xform->get_real_temp();
            } else {
               assert(!spanwise_xform);
               u_ptr = &output;
               if (output.data() == rhs.data())
                  f_ptr = twod_f;
               else
                  f_ptr = &rhs;
            }
            // Solve with 2D GMRES
            poisson_2d(*f_ptr,*u_ptr,gradient,type_x,(M+ls(j)*ls(j)*Jy*Jy),zbc_a,zbc_b,xbc_a,xbc_b);
            if (szy > 1) {
               // Assign the results back to the appropriate slice of the
               // spanwise transform temporary.  After all slabs are solved,
               // inverting the transform gives the real solution
               (*(spanwise_xform->get_real_temp()))(Range::all(),Range(j,j),Range::all()) =
                  *twod_u;
            }
         } else {
            /* If y has a complex transform (periodic), then we have both real and imaginary
               components to deal with.  In this case, we can solve for them one at a time,
               real part first.  We will wlays have a slab-copy to work with as well, since
               poisson_2d is specialized for real doubles */
            *twod_f = real((*(spanwise_xform->get_complex_temp()))
                  (Range::all(),Range(j,j),Range::all()));
            poisson_2d(*twod_f,*twod_u,gradient,type_x,(M+ls(j)*ls(j)*Jy*Jy),zbc_a,zbc_b,xbc_a,xbc_b);
            // Assign th solution in twod_u to the real part of the spanwise xform
            real((*(spanwise_xform->get_complex_temp()))(
                     Range::all(),Range(j,j),Range::all())) = *twod_u;
            // And repeat the above process with the imaginary part
            *twod_f = imag((*(spanwise_xform->get_complex_temp()))
                  (Range::all(),Range(j,j),Range::all()));
            poisson_2d(*twod_f,*twod_u,gradient,type_x,(M+ls(j)*ls(j)*Jy*Jy),zbc_a,zbc_b,xbc_a,xbc_b);
            imag((*(spanwise_xform->get_complex_temp()))(
                     Range::all(),Range(j,j),Range::all())) = *twod_u;
         }
      }
      // Now, the y-transformed solutoin should be in the spanwise_xform temporary.
      // Transform back to the destination array for physical space!
      if (spanwise_xform) {
         spanwise_xform->back_transform(&output,type_y);
         output = output/(spanwise_xform->norm_factor()); // Normalize
      } else {
//         twod_f = twod_u = 0;
      }
   }
      

   /* Specialization of Elliptic solver for x/y-trig, z-cheby expansion */
   void ElipSolver::chebz_solve(DTArray & rhs, DTArray & output,
         S_EXP type_x, S_EXP type_y, double bc_a, double bc_b) {
      /* Apply the forward transform */
      spec_transform->forward_transform(&rhs, type_x, type_y, NONE);
      int szx, szy, szz;
      int gcount = 0;
      spec_transform->get_sizes(szx,szy,szz);

      double Jz; DTArray * var_jac;
      gradient->get_jac(thirdDim,thirdDim,Jz,var_jac);
      double Lz = -2/Jz;

      if (spec_transform->use_complex()) {
         /* If either x or y is periodic, the output of the spectral transform
            is complex-valued */

         /* Because of the parallel splitting, the complex temporary is
            only guaranteed to have contiguous x/y planes.  The z-dimension
            may very well be split amongst the processors */
         CTArray & transxy = *(spec_transform->get_complex_temp());
         if (!cTransposer) {
            /* Allocate the required transposer */
            assert(!ctransyz);
            int csx, csy, csz=szz;
            if (type_y == FOURIER && szy > 1) {
               csy = szy/2 + 1;
               csx = szx;
            } else {
               assert(type_x == FOURIER && szx > 1);
               csx = szx/2 + 1;
               csy = szy;
            }
            cTransposer = new Transposer<complex<double> >(csx,csy,csz,firstDim,thirdDim);
            blitz::TinyVector<int,3> lbound, extent;
            blitz::GeneralArrayStorage<3> order;
            cTransposer->source_alloc(lbound,extent,order);
            ctransyz = new CTArray(lbound,extent,order);
         }
         CTArray & transyz = *ctransyz;
         cTransposer->back_transpose(transxy,transyz);
            

         Array<double,1> ks = spec_transform->wavenums(firstDim,
                                    transyz.lbound(firstDim),
                                    transyz.ubound(firstDim));
         Array<double,1> ls = spec_transform->wavenums(secondDim,
                                    transyz.lbound(secondDim),
                                    transyz.ubound(secondDim));

         /* Normalize to the defined grid.  The z-length is taken care of
            inside the GMRES solver, so look at only x/y here */

         double Jx, Jy;
         gradient->get_jac(firstDim,firstDim,Jx,var_jac);
         gradient->get_jac(secondDim,secondDim,Jy,var_jac);
         ks = -ks*ks*Jx*Jx;
         ls = -ls*ls*Jy*Jy;
//         if (type_x == FOURIER) ks = ks*4;
//         if (type_y == FOURIER) ls = ls*4;
         /* Loop over the x/y plane in spectral space */
         blitz::Array<double,1> resid(szz), soln(szz);
         for (int iii = transyz.lbound(firstDim); 
               iii <= transyz.ubound(firstDim); iii++) {
            for (int jjj = transyz.lbound(secondDim); 
                  jjj <= transyz.ubound(secondDim); jjj++) {
              /* The problem is separable in x/z and real/imaginary */
               resid=real(transyz(iii,jjj,blitz::Range::all()));
               int iter_count = 
                  poisson_1d(resid,soln,Lz,-(ks(iii)+ls(jjj)-M),bc_a,bc_a,bc_b,bc_b);
//               printf("(%d, %d: %f)r [%d]\n",iii,jjj,-(ks(iii)+ls(jjj)-M),iter_count);
//               cout << resid << soln;
               if (iter_count <= 0) {
                  fprintf(stderr,"ERROR: Solve (%d, %d: %f) real did not converge\nReturned GMRES iteration conut of %d\n",iii,jjj,-(ks(iii)+ls(jjj)-M),iter_count);
                  fprintf(stderr,"Maximum initial residual %g\n",max(abs(resid)));
                  assert(iter_count > 0);
               }
               gcount += iter_count;
               real(transyz(iii,jjj,blitz::Range::all()))=soln;
               
               resid=imag(transyz(iii,jjj,blitz::Range::all()));
               iter_count = 
                  poisson_1d(resid,soln,Lz,-(ks(iii)+ls(jjj)-M),bc_a,bc_a,bc_b,bc_b);
//               printf("(%d, %d: %f)i [%d]\n",iii,jjj,-(ks(iii)+ls(jjj)-M),iter_count);
//               cout << resid << soln;*/
               if (iter_count <= 0) {
                  fprintf(stderr,"ERROR: Solve (%d, %d: %f) imag did not converge\nReturned GMRES iteration cout of %d\n",iii,jjj,-(ks(iii)+ls(jjj)-M),iter_count);
                  assert(iter_count > 0);
               }
               gcount += iter_count;
               imag(transyz(iii,jjj,blitz::Range::all()))=soln;
            }
         }
         /* Now, back-transpose */
         cTransposer->transpose(transyz,transxy);
         
      } else {
         /* Real transform -- nearly identical to above, save using
            double rather than complex<double> */
         DTArray & transxy = *(spec_transform->get_real_temp());
         if (!rTransposer) {
            /* Allocate the required transposer */
            assert(!rtransyz);
            rTransposer = new Transposer<double>(szx,szy,szz,firstDim,thirdDim);
            blitz::TinyVector<int,3> lbound, extent;
            blitz::GeneralArrayStorage<3> order;
            rTransposer->source_alloc(lbound,extent,order);
            rtransyz = new DTArray(lbound,extent,order);

         }
         DTArray & transyz = *rtransyz;
         rTransposer->back_transpose(transxy,transyz);
            

         Array<double,1> ks = spec_transform->wavenums(firstDim,
                                    transyz.lbound(firstDim),
                                    transyz.ubound(firstDim));
         Array<double,1> ls = spec_transform->wavenums(secondDim,
                                    transyz.lbound(secondDim),
                                    transyz.ubound(secondDim));

         /* Normalize to the defined grid.  The z-length is taken care of
            inside the GMRES solver, so look at only x/y here */

         double Jx, Jy;
         gradient->get_jac(firstDim,firstDim,Jx,var_jac);
         gradient->get_jac(secondDim,secondDim,Jy,var_jac);
         ks = -ks*ks*Jx*Jx;
         ls = -ls*ls*Jy*Jy;
         /* Loop over the x/y plane in spectral space */
         blitz::Array<double,1> resid(szz), soln(szz);
         for (int iii = transyz.lbound(firstDim); 
               iii <= transyz.ubound(firstDim); iii++) {
            for (int jjj = transyz.lbound(secondDim); 
                  jjj <= transyz.ubound(secondDim); jjj++) {
              /* The problem is separable in x/z */
               resid=transyz(iii,jjj,blitz::Range::all());
//               fprintf(stderr,"Solving (%d,%d: %g [%g-%g]) rnorm %g\n",iii,jjj,M,bc_a,bc_b,max(abs(resid)));
               int iter_count = 
                  poisson_1d(resid,soln,Lz,-(ks(iii)+ls(jjj)-M),bc_a,bc_a,bc_b,bc_b);
               gcount += iter_count;
               if (iter_count <= 0) {
                  fprintf(stderr,"ERROR: Solve (%d, %d: %f) all-real did not converge\nReturned GMRES iteration cout of %d\n",iii,jjj,-(ks(iii)+ls(jjj)-M),iter_count);
                  assert(iter_count > 0);
               }
               transyz(iii,jjj,blitz::Range::all())=soln;
            }
         }
         /* Now, back-transpose */
         rTransposer->transpose(transyz,transxy);
      }
      /* Un-transform the solution (spectral space to physical space) */
      spec_transform->back_transform(&output, type_x, type_y, NONE);
      output = output / spec_transform->norm_factor(); // Normalize

//      int gcount_total = pssum(gcount);
//      if (master()) 
//         printf("%d total GMRES iterations\n",gcount_total);
      
   }
               

   
   /* Specialization of Ellptic solver for triply-spectral expansion */
   void ElipSolver::threespec_solve(DTArray & rhs, DTArray & output,
         S_EXP type_x, S_EXP type_y, S_EXP type_z) {
      /* Do the transform of our rhs to spectral space */
      spec_transform->forward_transform(&rhs, type_x, type_y, type_z);

      /* Get wavenumbers */
      Array<double,1> ks = spec_transform->wavenums(firstDim);
      Array<double,1> ls = spec_transform->wavenums(secondDim);
      Array<double,1> ms = spec_transform->wavenums(thirdDim);

      
      /* Normalize to our defined grid and square */
      /* If the transform is real, then the "default" grid is PI long, rather than 2*PI.
         This creates a small complication. */
      double Jx,Jy,Jz; DTArray * var_jac;
      gradient->get_jac(firstDim,firstDim,Jx,var_jac);
      gradient->get_jac(secondDim,secondDim,Jy,var_jac);
      gradient->get_jac(thirdDim,thirdDim,Jz,var_jac);
      ks = -ks*ks*Jx*Jx;
      ls = -ls*ls*Jy*Jy;
      ms = -ms*ms*Jz*Jz;
      
      // If any of the spectral expansions are sine, then we don't have a zero-frequency
      // to worry about.  (A sine expansion doesn't permit a constant mode)
      
      bool have_zero = !(type_x == SINE || type_y == SINE || type_z == SINE);
      
      if (spec_transform->use_complex()) { // Complex transform, so use that temporary
         CTArray & transformed = *(spec_transform->get_complex_temp());
         if (have_zero && M == 0.0f) { // Poisson case -- zero out zero frequency
            /* Note -- divide by zero gives floating point exceptions on some
               architectures/compile-time flags, so add a token value to send
               it to infinity or somesuch instead */
            transformed = transformed / (ks(ii) 
                  + ls(jj) 
                  + ms(kk) 
                  + 1e-12);
            if (!transformed.lbound(firstDim) && !transformed.lbound(secondDim)
                  && !transformed.lbound(thirdDim)) {
               /* Only zero out the 0 frequency if we have it on our array.
                  In the case of split arrays, this will only be true for
                  one processor */
               transformed(0,0,0) = 0;
            }
         } else {
            transformed = transformed / (ks(ii) + ls(jj) + ms(kk) - M);
         }
   //      *transformed = cast<double>((abs(*transformed) > 1e-12)) * (*transformed);

      } else {  // Strictly real transforms
         DTArray & transformed = *(spec_transform->get_real_temp());

         if (have_zero && M == 0.0f) { // Poisson case -- zero out zero frequency
            /* See complex zero-frequency case for discussion */
            transformed = transformed / (ks(ii) + ls(jj) + ms(kk) + 1e-12);
            if (!transformed.lbound(firstDim) && !transformed.lbound(secondDim)
                  && !transformed.lbound(thirdDim)) {
               /* Only zero out the 0 frequency if we have it on our array.
                  In the case of split arrays, this will only be true for
                  one processor */
               transformed(0,0,0) = 0;
            }
         } else {
            transformed = transformed / (ks(ii) + ls(jj) + ms(kk) - M);
         }
      }
      
      /* Now, transform back to real space (out) */
      spec_transform->back_transform(&output, type_x, type_y, type_z);
      output = output / (spec_transform->norm_factor()); // normalize
      
   }
   void ElipSolver::change_m(double newm) {
      M = newm;
      return;
   }
   
}
