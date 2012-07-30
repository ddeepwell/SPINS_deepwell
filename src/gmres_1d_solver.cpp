/* Wrapper for gmres and solver/preconditioning kernels, in order to
   provide a nice, simple, clean interface for solving the 1D
   Poisson/Helmholtz problems on a chebyshev grid.  Will need extension
   for Jacobian map */

// Include public interface
#include "gmres_1d_solver.hpp"

// And private interface
#include "gmres_1d_solver_impl.hpp"

#include "multigrid.hpp"

#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <map>
#include <vector>
using namespace std;


cheb_d2::cheb_d2(int sz, double l):
   band_d2(4,sz,blitz::fortranArray),
   pivot_factors(sz), factored(false),
   a_top(0), a_bot(0), b_top(0), b_bot(0), helm(0) {
      length = -l;
      size = sz;
      tline = new DTArray(1,1,size);
      chebformer = new Trans1D(1,1,size,thirdDim,Transformer::CHEBY,MPI_COMM_SELF);
}
void cheb_d2::set_bc(double h, double at, double ab,
      double bt, double bb) {
//   printf("Setting boundary conditions:\n");
//   printf("H = %g, a_top = %g, a_bot = %g\n",h,at,ab);
//   printf("b_top = %g, b_bot = %g\n",bt,bb);
   if (h != helm || at != a_top || ab != a_bot || 
         bt != b_top || bb != b_bot) {
      helm = h;
      a_top = at;
      a_bot = ab;
      b_top = bt;
      b_bot = bb;
      build_precond();
   }
}
cheb_d2::~cheb_d2() {
   delete tline;
   delete chebformer;
   for (vector<RType>::iterator i = r_free.begin(); i != r_free.end(); i++) {
      delete (*i)->zline;
      delete *i;
   }
   for (set<RType>::iterator i = r_used.begin(); i != r_used.end(); i++) {
      delete (*i)->zline;
      delete *i;
   }
   for (vector<BType>::iterator i = b_free.begin(); i != b_free.end(); i++) {
      delete (*i)->zline;
      delete *i;
   }
   for (set<BType>::iterator i = b_used.begin(); i != b_used.end(); i++) {
      delete (*i)->zline;
      delete *i;
   }
}
cheb_d2::RType cheb_d2::alloc_resid() {
   /* Check to see if we have any free resid vectors */
   gridline * r;
   if (r_free.size() == 0) {
      r = new gridline;
      r->zline = new DTArray(1,1,size);
   } else {
      r = r_free.back();
      r_free.pop_back();
   }
   r_used.insert(r);
   *r->zline = 0;
   r->extra = 0;
   return r;
}
cheb_d2::BType cheb_d2::alloc_basis() {
   gridline * r;
   if (b_free.size() == 0) {
      r = new gridline;
      r->zline = new DTArray(1,1,size);
   } else {
      r = b_free.back();
      b_free.pop_back();
   }
   b_used.insert(r);
   *r->zline = 0;
   r->extra = 0;
   return r;
}
void cheb_d2::free_resid(RType & r) {
   set<RType>::iterator loc = r_used.find(r);
   if (loc == r_used.end()) {
      fprintf(stderr,"ERROR: Freeing a GMRES residual that was not allocated\n");
      abort();
   } else {
      r_free.push_back(r);
      r_used.erase(loc);
   }
   r = 0;
   return;
}
void cheb_d2::free_basis(BType & b) {
   set<BType>::iterator loc = b_used.find(b);
   if (loc == b_used.end()) {
      fprintf(stderr,"ERROR: Freeing a GMRES residual that was not allocated\n");
      abort();
   } else {
      b_free.push_back(b);
      b_used.erase(loc);
   }
   b = 0;
   return;
}
void cheb_d2::matrix_multiply(BType & x, RType & r) {
   /* If we need normalization to remove a DC term, then calculate
      the x-sum */

   
//   printf("**m_mult(x)**\n"); 
//   x->print();
   if (a_top == 0 && a_bot == 0 && helm == 0) {
      r->extra = mean(*x->zline);
   }

   /* Take first derivative to the temporary array */
   deriv_cheb(*(x->zline),*chebformer,*tline);
   /* Scale by inverse length */
   *tline = (*tline) / (length/2);
   /* and second derivative to r->zline */
   deriv_cheb(*tline, *chebformer, *(r->zline));
//   deriv_cheb(*tline, *chebformer, *tline);
   /* And scale by inverse length */
   *(r->zline) = *(r->zline)/(length/2);


   /* Add any resulting Helmholtz term */
   if (helm != 0) {
      *(r->zline) = *(r->zline) - helm*(*(x->zline));
   } else if (a_top == 0 && a_bot == 0) {
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
//   printf("**m_mult(r)**\n"); 
//   r->print();
}
void cheb_d2::build_precond() {
   blitz::firstIndex ii;

   /* Build D2x via Jacobian on Dth, D2th 
    
      MATH NOTE:  This preconditioner is not the one that would
      result from a direct 3-point stencil discretization on the
      Cartesian-coordinate grid.  Instead, it is a -second order-
      stencil on the Theta-coordinate grid.  That is, functions
      that are linear or quadratic in theta get differentiated
      properly, everything else can go hang itself.

      The upshot is that in Cartesian space, this operator is exact
      for A + B*acos(x) + C*acos^2(x), which is a really weird
      basis (and typically really wrong near the endpoints for
      smooth functions).  As it turns out, this does not make a 
      large difference in -preconditioning- power.
    */
//   Array<double,1> theta(size,blitz::fortranArray);
   Array<double,1> x(size);
   Array<double,2> D(size,3), D2(size,3);
   band_d2 = 0;
   /* Theta goes from pi to 0 */
//   theta = M_PI - M_PI*(ii-1)/(size-1);
//   double dtheta = M_PI/(size-1);
   x = -length*cos(M_PI*ii/(size-1));
   get_fd_operator(x,D,D2);
   /* The third row of the matrix is the main diagonal.
      It's weird, but that's LAPACK for you. */
   for (int i = 1+band_d2.lbound(secondDim); i <= band_d2.ubound(secondDim)-1; i++) {
//      band_d2(3,i) = -2/dtheta/dtheta/(pow(sin(theta(i)),2)) - helm;
      band_d2(3,i) = D2(i-1,1)-helm;
   }
   /* BCs here are applcable for Dirichlet or Robin terms. */
//   band_d2(3,1) = a_bot+b_bot/(1-cos(M_PI/(size-1))); 
//   band_d2(3,size) = a_top+b_top/(1-cos(M_PI/(size-1)));
   band_d2(3,1) = a_bot+b_bot*D(0,1);
   band_d2(3,size) = a_top-b_top*D(size-1,1);
   /* Neumann BCs get faked.  This affects convergence, but MATLAB
      suggests that it's not overly bad. */
   if (a_top == 0 && a_bot == 0 && helm==0) {
      band_d2(3,1) += .01*b_bot;
      band_d2(3,size) += .01*b_top;
   }

   /* The fourth row is the subdiagonal */
   for (int i = 1; i <= size-2; i++) {
//      band_d2(4,i) =
//         -1/dtheta/2*cos(-dtheta+theta(i))/pow(sin(-dtheta+theta(i)),3) +
//         1/dtheta/dtheta/(pow(sin(theta(i)-dtheta),2));
      band_d2(4,i) = D2(i,0);
   }
//   band_d2(4,size-1) = -b_top/(1-cos(M_PI/(size-1))); 
   band_d2(4,size-1) = -b_top*D(size-1,0);
   band_d2(4,size) = 0;
   /* And the second is the superdiagonal */
   for (int i = 3; i <= size; i++) {
//   band_d2(2,i) =
//      1/dtheta/2*cos(theta(i)+dtheta)/pow(sin(theta(i)+dtheta),3) +
//      1/dtheta/dtheta/pow(sin(theta(i)+dtheta),2);
      band_d2(2,i) = D2(i-2,2);
   }
//   band_d2(2,2) = -b_bot/(1-cos(M_PI/(size-1))); 
   band_d2(2,2) = b_bot*D(0,2);
   band_d2(2,1) = 0;

   /* If we're faking pure-Neumann BCs, reset sub and superdiags */
/*   if (a_top == 0 && a_bot == 0 && helm==0) {
      band_d2(4,size-1) = band_d2(2,2) = 0;
   }*/

   /* Now, factor the preconditioner using dgbtrf */
   int n = size, info = 1, kl = 1, ku = 1, m = size,
       ldab = 4;
#if 0
   if (helm==0 && a_top == 0 && a_bot == 0) {
      cerr << D << band_d2;
      exit(1);
   }
#endif
   dgbtrf_(&m, &n, &kl, &ku, band_d2.data(), &ldab,
         pivot_factors.data(), &info);
   factored = true;
}
   

void cheb_d2::precondition(RType & r, BType & x) {
   blitz::firstIndex ii;
   /* Note that all this preconditioning should go into a seperate
      method, so that the matrix-building (and factorization!) is
      preserved.  This is less important now for a strictly tridiag
      system, but proper Robin/Neumann BCs -- especially with the
      bordered matrix representation, cry out for UMFPACK sparse
      factorization */
   /* Matrix defined, we can use it to solve the problem with
      dgbsv_ */
   assert(factored);
//   printf("**pc(r)**\n");
//   r->print();
   int info = 0, kl = 1, ku = 1, n = size, nrhs = 1,
       ldab = 4, ldb = size;
   /* Since the solve is destructive, copy over the zline */
   *(x->zline) = *(r->zline);

   if (a_top == 0 && a_bot == 0 && helm==0) {
      /* Estimate the compatibility condition and regularize the
         r-vector (to a very rough approximation) */
      x->extra = b_bot*((*r->zline)(0,0,0)) + 
                  b_top*((*r->zline)(0,0,size-1)) -
                  mean((*r->zline)(0,0,blitz::Range(1,size-2)));
      (*x->zline)(0,0,blitz::Range(1,size-2)) -= x->extra;
   } else {
      x->extra = 0;
   }

   /* And solve */
   dgbtrs_("N",&n, &kl, &ku, &nrhs, band_d2.data(), &ldab,
         pivot_factors.data(), x->zline->data(), &ldb, &info);
//   dgbsv_(&n, &kl, &ku, &nrhs, band_d2.data(), &ldab,
//         ipiv, (*x->zline).data(), &ldb, &info);
   /* If we have to do the DC-correction, make sure we apply t
      r->extra value such that mean(x->zline) == r->extra */
   if (a_top == 0 && a_bot == 0 && helm==0) {
      *x->zline = *x->zline - mean(*x->zline) + r->extra;
   }
//   printf("**pc(x)**\n");
//   x->print();
}

double cheb_d2::resid_dot(RType & r1, RType & r2) {
   const int lbound = r1->zline->lbound(thirdDim);
   const int ubound = r1->zline->ubound(thirdDim);
   const double extent = r1->zline->extent(thirdDim);
   double mysum = (*r1->zline)(0,0,lbound)*(*r2->zline)(0,0,lbound) +
                  (*r1->zline)(0,0,ubound)*(*r2->zline)(0,0,ubound);
   for (int i = r1->zline->lbound(thirdDim)+1; 
         i <= r1->zline->ubound(thirdDim)-1;
         i++) {
      mysum = mysum +
         (*r1->zline)(0,0,i)*(*r2->zline)(0,0,i)/extent;
   }
   if (a_top == 0 && b_top == 0 && helm == 0) {
      mysum += r1->extra*r2->extra;
   }
   return mysum;
//   if (a_top == 0 && b_top == 0 && helm == 0) {
//      return sum((*r1->zline)*(*r2->zline))/(r1->zline->extent(thirdDim))+
//         r1->extra*r2->extra;
//   } else {
//      return sum((*r1->zline)*(*r2->zline))/(r1->zline->extent(thirdDim));
//   }
}
double cheb_d2::basis_dot(BType & b1, BType & b2) {
   if (a_top == 0 && b_top == 0 && helm == 0) {
      return sum((*b1->zline)*(*b2->zline))/(b1->zline->extent(thirdDim))+
         b1->extra*b2->extra;
   } else {
      return sum((*b1->zline)*(*b2->zline))/(b1->zline->extent(thirdDim));
   }
}
void cheb_d2::resid_copy(RType & lhs, RType & rhs) {
   *(lhs->zline) = *(rhs->zline);
   lhs->extra = rhs->extra;
}
void cheb_d2::basis_copy(BType & lhs, BType & rhs) {
   *(lhs->zline) = *(rhs->zline);
   lhs->extra = rhs->extra;
}
void cheb_d2::resid_fma(RType & a, RType & b, double c) {
//   printf("**rfma(%g)**\n",c);
//   a->print();
//   b->print();
   *(a->zline) = *(a->zline) + c*(*b->zline);
   a->extra += c*(b->extra);
//   a->print();
}
void cheb_d2::basis_fma(BType & a, BType & b, double c) {
//   fprintf(stderr,"Building basis, %p %p %g\n",a,b,c);
//   fprintf(stderr,"zlines %p %p\n",a->zline,b->zline);
//   fprintf(stderr,"Datas %p %p\n",a->zline->data(),b->zline->data());
   *(a->zline) = *(a->zline) + c*(*b->zline);
//   fprintf(stderr,"Done");
   a->extra += c*(b->extra);
//   fprintf(stderr,"!\n");
}
void cheb_d2::resid_scale(RType & a, double c) {
   a->extra *= c;
   *a->zline = c*(*a->zline);
}
void cheb_d2::basis_scale(BType & a, double c) {
   a->extra *= c;
   *a->zline = c*(*a->zline);
}

bool cheb_d2::noisy() const {
   return false;
}

int poisson_1d(Array<double,1> & resid, Array<double,1> & soln,
      double length, double helmholtz, double a_top, 
      double a_bot, double b_top, double b_bot) {
   /* Solves the 1D Poisson/Helmholtz problem on an Array<double,1>
      using the preconditioned GMRES algorithm with the abovve
      kernel. */

   /* In the interests of a quick-and-dirty solution, we'll allocate
      the kernel and solver here as static variables and error out
      if N ever happens to change.  There's no need to expose the
      internals of GMRES or the kernel to the rest of the application */
   static int N = 0;
   static double len = 0;
   static cheb_d2::RType resid_vec = 0;
   static cheb_d2::BType ans_vec = 0;
   static cheb_d2 * kernel = 0;
   static GMRES_Solver<cheb_d2> * gmresser = 0;
   /* If the residual is zero, the solution is obviously itself zero */
//   fprintf(stderr,"Calling poisson_1d with parameters:\n");
//   fprintf(stderr,"length=%f, helm=%f, a_top=%f, a_bot=%f\n",length,helmholtz,a_top,a_bot);
//   fprintf(stderr,"b_top=%f, b_bot=%f\n",b_top,b_bot);
//   fprintf(stderr,"Array data pointers resid=%p, soln=%p\n",resid.data(),soln.data());
//   cerr << resid;
   if (all(resid == 0)) {
      soln = 0;
      return 1;
   }
   if (N==0) {
      N = resid.extent(blitz::firstDim);
      len = length;
      kernel = new cheb_d2(N,length);
      resid_vec = kernel->alloc_resid();
      ans_vec = kernel->alloc_basis();
      gmresser = new GMRES_Solver<cheb_d2>(kernel);
   } else {
      assert (N == resid.extent(blitz::firstDim));
      assert (len == length);
   }

   if (kernel->noisy()) {
      fprintf(stderr,"Solving 1D GMRES (cheby) problem, with parameters:\n   %g, (%g-%g), (%g-%g)\n",helmholtz,a_top,b_top,a_bot,b_bot);
   }
   kernel->set_bc(helmholtz,a_top,a_bot,b_top,b_bot);
   *(resid_vec->zline) = resid(blitz::tensor::k);
//   cout << "lb:" << resid_vec->zline->lbound() << "-" << 
//      ans_vec->zline->lbound() << endl <<
//      "st:" << resid_vec->zline->stride() << "-" <<
//      ans_vec->zline->stride() << endl;
   resid_vec->extra = 0;
   /* Perform the GMRES step, to a tight tolerance */


   double final_error=0;
   int itc;
   itc = gmresser->Solve(ans_vec,resid_vec,1e-7,20,3,&final_error);
   soln = (*(ans_vec->zline))(0,0,blitz::Range::all());
   if (itc < 0) {
      /* Convergence problems */
      fprintf(stderr,"Warning: GMRES not converged (%d)\n",-itc);
      fprintf(stderr,"Problem parameters (%d,%g), (%g,%g)x(%g,%g) + %g\n",N,length,a_top,b_top,a_bot,b_bot,helmholtz);
      fprintf(stderr,"Starting vector norm of %g\n",sqrt(kernel->resid_dot(resid_vec,resid_vec)));
      fprintf(stderr,"Candidate (nonconverged) solution of norm %g\n",sqrt(kernel->basis_dot(ans_vec,ans_vec)));
      fprintf(stderr,"Converged to relative error %g\n",final_error);
   }
   return (itc);
}
