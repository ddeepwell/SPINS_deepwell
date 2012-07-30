#ifndef GMRES_HPP
#define GMRES_HPP 1
#include <vector>
#include <blitz/array.h>
#include <stdio.h>
#include <iostream>
//#include <mkl_lapack.h>
/* LAPACK function prototype */
extern "C" {
   void dgels_(char *, int *, int *, int *, double *, int *, double *,
               int *, double *, int *, int *);
   void dgelsd_(int * M, int * N, int * NRHS, double * A, int * LDA, double * B,
               int * LDB, double * S, double * RCOND, int * RANK, double * WORK,
               int * LWORK, int * IWORK, int * INFO);
}
/* gmres.hpp -- header class for gmres template class, which abstracts out the matrix-vector
   multiply, preconditioning, and dot products of GMRES to a user-specified type conforming
   to a provided interface.

   Note that as per LAPACK conventions, the Blitz arrays used in this module are
   FORTRAN-style arrays; they are stored column-major and indexed from one, rather
   than zero */

/* Interface class for the GMRES module.  In order to allow inheritance from this
   interface, we have to have the relevant type specifiers as template parameters.
   Other options (for example using void *) mean that the proper implementation
   is restricted in type spefications for compatibility */
template <class resid_type, class basis_type> class GMRES_Interface {
   public:
      /* GMRES module does not do allocation of anything by itself.  For simple, low-
         dimension problems, it might make sense to pass around a full array.  For
         complicated problems where the relevant vectors might be split over many
         processors in parallel, allocation might be equally complicated.  So the
         right answer is to punt it all to the controlling module */
      typedef resid_type RType;
      typedef basis_type BType;

      virtual resid_type alloc_resid()=0;
      virtual basis_type alloc_basis()=0;
      virtual void free_resid(resid_type &)=0;
      virtual void free_basis(basis_type &)=0;

      /* The required matrix-vector operations are the multiply and preconditioning.
         In reality, GMRES will work perfectly well on an infinite-dimension functional
         space.  But since it is almost always introduced as an iterative -matrix- solver,
         using that vocabulary is most sensible */
      
      virtual void matrix_multiply(basis_type & x, resid_type & b)=0;

      /* And generally speaking, GMRES is used with a preconditioner -- an approximate
         inverse of the full problem */

      virtual void precondition(resid_type & r, basis_type & x)=0;
      
      /* Lower-level functions are the dot products ... */
      virtual double resid_dot(resid_type & r1, resid_type & r2)=0;
      virtual double basis_dot(basis_type & x1, basis_type & x2)=0;

      /* Assignment (copy) */
      virtual void resid_copy(resid_type & lhs, resid_type & rhs) = 0 ;
      virtual void basis_copy(basis_type & lhs, basis_type & rhs) = 0 ;

      /* Scaled addition, called "fma" after "fused multiply add".  To keep 
         allocation completely explicit, this is written as:
            A += c*B, with a call
         foo_fma(A,B,c) */
      virtual void resid_fma(resid_type & a, resid_type & b, double c=1)=0;
      virtual void basis_fma(basis_type & a, basis_type & b, double c=1)=0;

      /* And finally scaling (multiplication/division) */
      virtual void resid_scale(resid_type & a, double c) = 0;
      virtual void basis_scale(basis_type & a, double c) = 0;

      virtual bool noisy() const { return false; };
      
};
template <class Controller> class GMRES_Solver {
   /* Class to solve a linear problem A*x=b using GMRES.  Rather than rely on
      a strict matrix formulation, GMRES can abstract out the linear operations,
      and we do that here to an instance of the Controller class.  The interface
      assumed by a Controller is specified in GMRES_Interface above. */
   public:
      typedef typename Controller::RType RType;// Residual type
      typedef typename Controller::BType BType;// Basis type
      
      Controller * ops; // Contains matrix/vector, vector/vector, and scalar/vector ops

      GMRES_Solver(Controller * operators,double prec=1e-8,
                   int inner=10, int outer=10 ): // Constructor
         ops(operators), default_inner(inner), 
         default_outer(outer), default_prec(prec) {
            lapack_workspace = 0;
            lwork_size = -1;
      }

      ~GMRES_Solver() { // Destructor
         free_workspace();
      }

      /* Solve the problem of A*x = b, performing outer iterations
        until the remaining residual has relative L2 norm less than prec. */
      int Solve(BType & x, RType & b, double prec = -1,
            int inner = -1, int outer = -1, double * err_out = 0) {
         using std::cout; using std::endl;
         //cout << b;
         inner = (inner <= 0) ? default_inner : inner;
         outer = (outer <= 0) ? default_outer : outer;
         prec = (prec <= 0) ? default_prec : prec;
         if (inner != last_inner) {
            /* If we've changed the number of inner iterations, the workspace
               needed for the least-squares solve may also change.  To be safe,
               delete the workspace and re-allocate it as the first thing in
               an inner iteration */
            free_workspace();
            last_inner = inner;
         }
         RType rnew = ops->alloc_resid(); // Get vector for remaining residual
         RType rapply = ops->alloc_resid(); // Vector for applying M*x
         BType xinner = ops->alloc_basis(); // Vector for result of inner gmres

         /* Copy b to rnew, since we have a zero initial guess */
        ops->resid_copy(rnew,b); // rnew = b
        ops->basis_scale(x,0);


        double start_norm = sqrt(ops->resid_dot(b,b));
        double error_norm = -1;
        if (ops->noisy())
           fprintf(stderr,"Solving GMRES problem to tolerance %g with %d inner, %d outer iterations.\nStarting norm: %e\n",prec,inner,outer,start_norm);

        int innerits = 0; // count number of inner iterations
        for (int i = 0; i < outer; i++) {
           /* Call the inner iteration */
           innerits += gmres_inner(rnew, xinner, inner, prec*start_norm); 
           if (i != 0) {
              ops->basis_fma(x,xinner,1); // Add the result to x
           } else {
              ops->basis_copy(x,xinner);
           }
           ops->matrix_multiply(x,rapply); // M*x = Rapply
           ops->resid_copy(rnew,b); /* Rnew = b */
           ops->resid_fma(rnew,rapply,-1); /* Rnew = b - A*x */
           error_norm = sqrt(ops->resid_dot(rnew,rnew));
           if (ops->noisy())
              fprintf(stderr,"Outer iteraiton %d with absolute norm %g\n",i+1,error_norm);
//           cout << error_norm << endl;
           if ((error_norm / start_norm) < prec) { // converged
              /* Free temporaries */
              ops->free_resid(rnew);
              ops->free_resid(rapply);
              ops->free_basis(xinner);
              /* Return the total number of iterations taken */
              if (err_out) {
                 *err_out = error_norm/start_norm;
              }
              return(innerits);
           }
        }
        /* Free temporaries */
        ops->free_resid(rnew);
        ops->free_resid(rapply);
        ops->free_basis(xinner);
        /* Not converged.  Return the -negative- of total iterations */
        if (err_out) {
           *err_out = error_norm/start_norm;
        }
        if (innerits < inner*outer) {
           /* Converged in the inner iteration with LAPACK-estimated
              error, but nonlinearity in the operator screwed us up.

              Generally, this error is tolerable. */
//           fprintf(stderr,"WARNING: (%s:%d), only converged to %g of %g (%d/%d its)\n",
//                 __FILE__,__LINE__,error_norm/start_norm,prec,innerits,inner);
           return(innerits);
        }
        return(-innerits);
      }

   private:
      double * lapack_workspace; // Work array for LAPACK routines
      int    * lapack_iworkspace; // Integer work array for "
      int      lwork_size;    // Size of the LAPACK workspace array
      int      default_inner; // Default number of inner iterations
      int      last_inner;    // Number of inner iterations for last solve
      int      default_outer; // Default limit of outer iterations
      double   default_prec;  // Default precision limit for solve

      void free_workspace() {
         if (lapack_workspace) {
            delete[] lapack_workspace;
            lapack_workspace = 0;
         }
         lwork_size = -1;
      }

      int gmres_inner(RType & start_r, BType & out_x, int num_its, double prec) {
         /* Iterate via the GMRES inner iteration method for num_its or until
            the problem (A*x = start_r) is solved to prec precision */
         using std::cout; using std::endl; using std::cerr;
/*         cout << start_r << endl << num_its << endl << prec << endl;
         cout << "---- starting gmres_inner ----\n";*/
         using blitz::Array; using blitz::fortranArray;
         using std::vector;
         /* Hessenberg matrix for the Arnoldi iteration, and a copy since LAPACK
            overwrites it */
         Array<double,2> hess(num_its+1,num_its,fortranArray),
            hess_copy(num_its+1,num_its,fortranArray);
         /* RHS vector for the least-squares problem, overwritten with the answer */
         Array<double,1> rhs_vec(num_its+1,fortranArray);
         /* SVD vector for the singular values of the hessian */
         Array<double,1> svd_vec(num_its+1,fortranArray);
         /* Constants for the LAPACK call */
         int LDA = num_its+1, LDB = num_its + 1, NRHS = 1, INFO = 0;
         int RANK;
         double RCOND = 1e-6;

         if (!lapack_workspace) {
            blitz::firstIndex ii; blitz::secondIndex jj;
            /* No workspace defined, so make a workspace-request call to DGELS in LAPACK */
            int M = num_its+1, 
                N = num_its; 
            double workzero;
            /* Set A to an identity matrix + 1 in lower-right-hand corner */
            hess = (ii == jj) || (ii-1 == jj) || (ii+1 == jj);
            //            hess(num_its+1,num_its) = 1;
            /* Set B to e0 */
            rhs_vec = 0; rhs_vec(1) = 1;
            /* Call to dgels, LAPACK, for workspace query */
//            dgels_("N",&M,&N,&NRHS,hess.data(),&LDA,rhs_vec.data(),
//                  &LDB,&workzero,&lwork_size,&INFO);
            dgelsd_(&M,&N,&NRHS,hess.data(),&LDA,rhs_vec.data(),
                  &LDB, svd_vec.data(), &RCOND, &RANK, &workzero,
                 &lwork_size,lapack_iworkspace,&INFO); 
//            fprintf(stderr,"Got work size %f\n",workzero);
            /* Allocate the workspace */
            lapack_workspace = new double[int(workzero)];
            lapack_iworkspace = new int[11*M+3*int(1+log2(double(M)))*M];
            lwork_size = int(workzero);
         }
         hess = 0;

         /* Perform the inner iteration */
         /*  First, allocate vectors for residuals and the Krylov basis */
         blitz::Vector<RType> resid_vec(num_its+1);
         blitz::Vector<BType> basis_vec(num_its); // Note, these are base 0
         for (int k = 0; k < num_its; k++) {
            resid_vec(k) = ops->alloc_resid();
            basis_vec(k) = ops->alloc_basis();
         }
         resid_vec(num_its) = ops->alloc_resid();

         /* Compute the norm of our starting r */
         double starting_norm = sqrt(ops->resid_dot(start_r,start_r));
         /* And scale for assignment to the residual vector */
         ops->resid_copy(resid_vec(0),start_r);
         ops->resid_scale(resid_vec(0),1/starting_norm);


         int my_it = 1;
         double remaining_error = 1;
         for (my_it = 1; my_it <= num_its; my_it++) {
            /* Find an x vector that approximately solves the prior basis */
            ops->precondition(resid_vec(my_it-1), basis_vec(my_it-1));
            /* Apply the operator the current basis vector */
            ops->matrix_multiply(basis_vec(my_it-1),resid_vec(my_it));
//            cout << resid_vec[my_it];
            /* Grahm-Schmidt orthogonalization */
            for (int k = 1; k <= my_it; k++) {
               double dot = ops->resid_dot(resid_vec(k-1),resid_vec(my_it));
               //cout << "**" << dot << "**\n";
               /* Assign the dot to the appropriate row (k) / column (my_it) of
                  the Hessenberg matrix */
               hess(k,my_it) = dot;
               /* And subtract dot*resid_vec[k-1] from the computed residual */
               ops->resid_fma(resid_vec(my_it),resid_vec(k-1),-dot);
               //cout << resid_vec[my_it];
            }
            /* Now, normalize the remainder */
            double norm = sqrt(ops->resid_dot(resid_vec(my_it),resid_vec(my_it)));
//            fprintf(stderr,"norm %g\n",norm);
            ops->resid_scale(resid_vec(my_it),1/norm);
            //cout << resid_vec[my_it];
            hess(my_it+1,my_it) = norm;

            /* Now, solve the least-squares problem with LAPACK */
            {
               /* LAPACK overwrites the matrix, so copy */
//               std::cerr.precision(8);
//               std::cerr.width(8);
//               std::cerr.setf( std::ios::scientific );
//               std::cerr << hess;
               hess_copy = hess;
               int M = my_it+1, N = my_it;
               rhs_vec = 0; rhs_vec(1) = 1;
               /* Make the LAPACK call */
//               fprintf(stderr,"M: %d N: %d\n",M,N);
//            fprintf(stderr,"Args m %d n %d nrhs %d lda %d ldb %d\n",M,N,NRHS,LDA,LDB); fflush(stderr);
            //std::cerr << hess_copy; std::cerr.flush();
//               dgels_("N",&M,&N,&NRHS,hess_copy.data(),&LDA,
//                     rhs_vec.data(), &LDB, lapack_workspace, &lwork_size,
//                     &INFO);
               dgelsd_(&M,&N,&NRHS,hess_copy.data(),&LDA,rhs_vec.data(),
                     &LDB, svd_vec.data(), &RCOND, &RANK, lapack_workspace,
                    &lwork_size,lapack_iworkspace,&INFO); 
//               fprintf(stderr,"%s:%d RANK %d\n",__FILE__,__LINE__,RANK); fflush(stderr);
               /* rhs_vec.data() contains the answer and remainder */
//               std::cerr << svd_vec;
//               std::cerr << rhs_vec;
               remaining_error = fabs(rhs_vec(my_it+1));
            }
//            if (abs(rhs_vec(my_it)) < 1e-14*max(abs(rhs_vec))) {
               /* If the last rhs vector is much smaller than the maximum,
                  we've either stagnated with convergence or we're running into
                  rounding error.  Either way, there's probably no progress left
                  to make in double precision. */
/*               cout << "Terminating early with near-zero change (" <<
                  abs(rhs_vec(my_it)) << "/" << max(abs(rhs_vec)) << ")" << endl;*/
//               my_it++; break;
//            }
//            cout << hess;
            if (ops->noisy())
               fprintf(stderr,"  %d: Remaining error %g\n",my_it,remaining_error);
//           printf("Remaining error (LAPACK): %g\n",remaining_error*starting_norm); 
            if (remaining_error < 0.1*prec/starting_norm && my_it < num_its-1) {
               /* Satisfied the inner iteration to the desired precision.
                  Note that the residual error returned by LAPACK is sometimes
                  off by a fair bit.  If things worked really well, we'd
                  just need to test remaining_error < prec.  Instead,
                  include a safety factor of 10.  If this doesn't work out,
                  then we'll have to verify by actually constructing the
                  solution x and testing error with more matrix multiplies. */
               /* Build the output x */
               my_it++; break;
#if 0
               build_x(out_x, basis_vec, rhs_vec, starting_norm, my_it+1);
               /* Mutply M*x to find r */
               ops->matrix_multiply(out_x,resid_vec(my_it+1));
               /* Subtract the r we're supposed to solve for */
               ops->resid_fma(resid_vec(my_it+1),start_r,-1);
               double normi = sqrt(ops->resid_dot(resid_vec(my_it+1),resid_vec(my_it+1)));
//               printf("Checking for early termination, error %g\n",normi);
               if (normi < prec) {
                  
//               cout << "(" << remaining_error << ")\n";
                  my_it++;
                  break;
               }
#endif
            }
         }
//         cout << hess;
         /* Build the output x */
         build_x(out_x, basis_vec, rhs_vec, starting_norm, my_it);
         /* Free all allocated basis/residual vectors */
         for (int k = 0; k < num_its; k++) {
             ops->free_resid(resid_vec(k));
             ops->free_basis(basis_vec(k));
         }
         ops->free_resid(resid_vec(num_its));
//         cout << "Ending inner iteration " << my_it - 1 <<
//                  " with error " << remaining_error*starting_norm << endl;
         return my_it-1;
      }

      void build_x(BType out_x, blitz::Vector<BType> & basis_vec,
            blitz::Array<double,1> & rhs_vec, double starting_norm,
            int my_it) {
         /* Now, the Krylov basis is in basis_vec, and the proper multiples are in
            rhs_vec. */
//         cout << out_x;
         ops->basis_scale(out_x,0);
//         cout << out_x;
         /* This loop runs BACKWARDS.  In general, when we're solving a problem well
            the magnitudes decay pretty damn fast.  Adding from one to the end
            can easily mean we trash the entire contribution from the last few
            terms by rounding error.  We really should sort this vector and sum
            from smallest to largest magnitude, but going from end to start
            should also do the trick */
         for (int k = my_it-1; k > 0; k--) {
            ops->basis_fma(out_x,basis_vec(k-1),rhs_vec(k));
         }
         /* And finally, scale by the starting norm */
         ops->basis_scale(out_x,starting_norm);
      }

};
#endif      
