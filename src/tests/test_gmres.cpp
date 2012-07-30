#include <mpi.h>
#include "../TArray.hpp"
#include "../T_util.hpp"
#include "../gmres.hpp"
#include "../Parformer.hpp"
#include <blitz/array.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
/* Tests GMRES for a single-dimension */

using namespace std;
using TArrayn::DTArray;
using Transformer::Trans1D;
using TArrayn::deriv_cheb;
using TArrayn::firstDim; using TArrayn::secondDim; using TArrayn::thirdDim;

extern "C" {
   extern void dgbsv_(int *, int *, int *, int *, double *, 
               int *, int *, double *, int *, int *);
}

struct gridline { 
   /* Struct to hold together the main vector itself (zline, after the typical
      dimension this will be used with) and an extra double for normalization
      with Neumann-Neumann BCs */
   DTArray * zline;
   double extra;
};

class cheb_d2 : public GMRES_Interface<gridline *, gridline *> {
   private:
      int size;
      DTArray * tline;
      Trans1D * chebformer;

      /* BC values */
      double a_top, a_bot, b_top, b_bot;
      /* Helmholtz parameter */
      double helm;

      /* Length (eventually extend to Jacobian) */
      double length;
   public:
      cheb_d2(int sz, double atop=1, double abot=1,
            double btop=0, double bbot=0) {
         b_top = btop;
         b_bot = bbot;
         a_top = atop;
         a_bot = abot;
         length = -1;
         size = sz;
         tline = new DTArray(1,1,size);
         chebformer = new Trans1D(1,1,size,thirdDim,Transformer::CHEBY);
      }
      ~cheb_d2() {
         delete tline;
         delete chebformer;
      }
      RType alloc_resid() {
//         cout << "+r" << endl;
         gridline * r = new gridline;
         r->zline = new DTArray(1,1,size);
         *r->zline = 0;
         r->extra = 0;
         return r;
      }
      BType alloc_basis() {
         //cout << "+b" << endl;
         gridline * r = new gridline;
         r->zline = new DTArray(1,1,size);
         *r->zline = 0;
         r->extra = 0;
         return r;
      }

      void free_resid(RType & r) {
         //cout << "-r" << endl;
         delete r->zline;
         delete r;
         r = 0;
      }
      void free_basis(BType & b) {
         //cout << "-b" << endl;
         delete b->zline;
         delete b;
         b = 0;
      }
      
      void matrix_multiply(BType & x, RType & r) {
         /* If we need normalization to remove a DC term, then calculate
            the x-sum */
         if (a_top == 0 && a_bot == 0 && helm == 0) {
            r->extra = sum(*x->zline);
         }
         
         /* Take first derivative to the temporary array */
         deriv_cheb(*(x->zline),*chebformer,*tline);
         /* Scale by inverse length */
         *tline = (*tline) / (length);
//         cout << *tline;
         /* And second derivative to r->zline */
         deriv_cheb(*tline, *chebformer, *(r->zline));
         /* And scale by inverse length */
         *(r->zline) = *(r->zline)/(length);


         /* Add any resulting Helmholtz term */
         if (helm != 0) {
            *(r->zline) = *(r->zline) - helm*(*(x->zline));
         } else {
            /* If we're removing the DC term, add the inconsistency term
               from x */
            *(r->zline) = *(r->zline) + x->extra;
         }
         /* BCs */
         int end = r->zline->ubound(thirdDim);
         /* Reminder -- domain goes from bottom (0) to top (end).  Chebyshev
            discretizations don't normally do this, so cheby-domains will essentially
            have negative lengths.  Going to have to write a "grid generator" that
            remembers this, because I won't. */
         (*r->zline)(0,0,0) = a_bot*((*x->zline)(0,0,0)) - b_bot*((*tline)(0,0,0));
         (*r->zline)(0,0,end) = a_top*((*x->zline)(0,0,end)) +
                                b_top*((*tline)(0,0,end));
//         cout << "MATMULT" << endl << *x->zline << *r->zline <<
//            "................" << endl;
      }
      void precondition(RType & r, BType & x) {
         blitz::firstIndex ii;
         /* Note that all this preconditioning should go into a seperate
            method, so that the matrix-building (and factorization!) is
            preserved.  This is less important now for a strictly tridiag
            system, but proper Robin/Neumann BCs -- especially with the
            bordered matrix representation, cry out for UMFPACK sparse
            factorization */
         /* Non-null preconditioner */
         /* Build D2x via Jacobian on Dth, D2th */
         Array<double,2> band_d2(4,size,blitz::fortranArray);
         Array<double,1> theta(size,blitz::fortranArray);
         /* Theta goes from pi to 0 */
         theta = M_PI - M_PI*(ii-1)/(size-1);
         double dtheta = M_PI/(size-1);
         /* The third row of the matrix is the main diagonal.
            It's weird, but that's LAPACK for you. */
         band_d2(3,blitz::Range::all()) =
            -2/dtheta/dtheta/(pow(sin(theta(ii)),2));
         /* BCs here are applcable for Dirichlet or Robin terms. */
         band_d2(3,1) = a_bot+b_bot*(1-cos(M_PI/(size-1))); 
         band_d2(3,size) = a_top+b_top*(1-cos(M_PI/(size-1)));
         /* Neumann BCs get faked.  This affects convergence, but MATLAB
            suggests that it's not overly bad. */
         if (a_top == 0 && a_bot == 0 && helm==0) {
            band_d2(3,1) = 1;
            band_d2(3,size) = 1;
         }

         /* The fourth row is the subdiagonal */
         band_d2(4,blitz::Range::all()) =
            -1/dtheta/2*cos(-dtheta+theta(ii))/pow(sin(-dtheta+theta(ii)),3) +
            1/dtheta/dtheta/(pow(sin(theta(ii)-dtheta),2));
         band_d2(4,size-1) = -b_top*(1-cos(M_PI/(size-1))); 
         band_d2(4,size) = 0;
         /* And the second is the superdiagonal */
         band_d2(2,blitz::Range::all()) =
            1/dtheta/2*cos(theta(ii)+dtheta)/pow(sin(theta(ii)+dtheta),3) +
            1/dtheta/dtheta/pow(sin(theta(ii)+dtheta),2);
         band_d2(2,2) = -b_bot*(1-cos(M_PI/(size-1))); 
         band_d2(2,1) = 0;

         /* If we're faking pure-Neumann BCs, reset sub and superdiags */
         if (a_top == 0 && a_bot == 0 && helm==0) {
            band_d2(4,size-1) = band_d2(2,2) = 0;
         }

         /* Matrix defined, we can use it to solve the problem with
            dgbsv_ */
         int info = 0, kl = 1, ku = 1, n = size, nrhs = 1,
             ldab = 4, ipiv[size], ldb = size;
         /* Since the solve is destructive, copy over the zline */
         *(x->zline) = *(r->zline);
         (x->extra) = (r->extra);

         /* And solve */
         dgbsv_(&n, &kl, &ku, &nrhs, band_d2.data(), &ldab,
               ipiv, (*x->zline).data(), &ldb, &info);
      }

      double resid_dot(RType & r1, RType & r2) {
         if (a_top == 0 && b_top == 0 && helm == 0) {
            return sum((*r1->zline)*(*r2->zline))/(r1->zline->extent(thirdDim))+
                  r1->extra*r2->extra;
         } else {
            return sum((*r1->zline)*(*r2->zline))/(r1->zline->extent(thirdDim));
         }
      }
      double basis_dot(BType & b1, BType & b2) {
         if (a_top == 0 && b_top == 0 && helm == 0) {
            return sum((*b1->zline)*(*b2->zline))/(b1->zline->extent(thirdDim))+
               b1->extra*b2->extra;
         } else {
            return sum((*b1->zline)*(*b2->zline))/(b1->zline->extent(thirdDim));
         }
      }
      void resid_copy(RType & lhs, RType & rhs) {
         *(lhs->zline) = *(rhs->zline);
         lhs->extra = rhs->extra;
      }
      void basis_copy(BType & lhs, BType & rhs) {
         *(lhs->zline) = *(rhs->zline);
         lhs->extra = rhs->extra;
      }

      void resid_fma(RType & a, RType & b, double c) {
         *(a->zline) = *(a->zline) + c*(*b->zline);
         a->extra += c*(b->extra);
      }
      void basis_fma(BType & a, BType & b, double c) {
         *(a->zline) = *(a->zline) + c*(*b->zline);
         a->extra += c*(b->extra);
      }
      void resid_scale(RType & a, double c) {
         a->extra *= c;
         *a->zline = c*(*a->zline);
      }
      void basis_scale(BType & a, double c) {
         a->extra *= c;
         *a->zline = c*(*a->zline);
      }
};

blitz::thirdIndex kk;
blitz::secondIndex jj;
blitz::firstIndex ii;
int main(int argc, char ** argv) {
   MPI_Init(&argc,&argv);
   int N = 0;
   if (argc > 1) N = atoi(argv[1]);
   cout << "N = " << N << endl;
   if (N <= 0) N = 32;
   Array<double,1> x(N);
   x = -cos(M_PI*ii/(N-1));
   for (int k = 0; k <= 2; k++) {
      /* To test Dirichlet, Neumann, and Robin BCs, step through the possible
         a/b combinations:
         (1,0) - (1,1)/sqrt(2) - (0,1)
         */
      double dbc = (2-k)/sqrt(4-4*k+2*k*k); 
      double nbc = k/sqrt(4-4*k+2*k*k);
      cheb_d2 a(N,dbc,dbc,nbc,nbc); // Crete the operator-kernel
      printf("Testing bc pair (%.2f, %.2f)...\n",dbc,nbc);
      gridline * init_r = a.alloc_resid();
      gridline * final_z = a.alloc_basis();
      /* Create the rue solution, sin(x) */
      Array<double,1> true_soln(x);
      /* The second derivative of sin(x) is -sin(x) */
//      *init_r->zline = -sin(x(kk));
      *init_r->zline = 0;
      /* BC's -- general Robin formulation.  The normal is an outward-facing
         normal, so the -1BC gets a -b */
/*      (*init_r->zline)(0,0,0) = dbc*sin(-1) - nbc*cos(-1);
      (*init_r->zline)(0,0,N-1) = dbc*sin(1) + nbc*cos(1);*/
      (*init_r->zline)(0,0,0) = dbc*(-1) - nbc*(1);
      (*init_r->zline)(0,0,N-1) = dbc*(1) + nbc*(1);
      GMRES_Solver<cheb_d2> solver(&a);
      int itercount = solver.Solve(final_z,init_r,1e-8,10,10);
      /* Now, hopefully all goes well.  */
      printf("Solved in %d iterations. Maximum error %g\n",
            itercount, max(abs((*final_z->zline)(ii,jj,kk)-true_soln(kk))));
      a.free_resid(init_r);
      a.free_basis(final_z);
   }
   MPI_Finalize();
}
