#include <blitz/array.h>
#include "TArray.hpp"
#include "T_util.hpp"
#include "grad.hpp"
#include "Parformer.hpp"
#include "gmres_2d_solver.hpp"
#include "Par_util.hpp"

// private interface for gmres-solving class
#include "gmres_2d_solver_impl.hpp"

using namespace Transformer;
using namespace TArrayn;

using std::vector;
using std::set;
using blitz::Range;

#define SYNC(__x__) { int myrank, nproc; MPI_Comm_size(my_comm,&nproc); MPI_Comm_rank(my_comm,&myrank); \
                     if (master(my_comm)) fprintf(stderr,"SYNC: %s:%d\n",__FILE__,__LINE__); \
                     for (int _i = 0; _i < nproc; _i++) { \
                     if (myrank == _i) { \
                        __x__; \
                     } MPI_Barrier(my_comm); } }

Cheb_2dmg::Cheb_2dmg(int size_x, int size_z, MPI_Comm c):
   szx(size_x), szz(size_z), my_comm(c), temp1(0), temp2(0),
   temp3(0), indefinite_problem(false), dir_zbc(0), neum_zbc(0),
   dir_xbc(0), neum_xbc(0),
   helm(0), grad_3d(0), grad_2d(0), mg_precond(0),
   dx_top(split_range(szx,c)), dz_top(split_range(szx,c)),
   dx_bot(split_range(szx,c)), dz_bot(split_range(szx,c)),
   dx_left(szz), dx_right(szz), dz_left(szz), dz_right(szz) {
      temp1 = alloc_array(szx,1,szz,my_comm);
      temp2 = alloc_array(szx,1,szz,my_comm);
      temp3 = alloc_array(szx,1,szz,my_comm);
      r2d = new Array<double,2> (split_range(szx,my_comm),Range(0,szz-1));
      s2d = new Array<double,2> (split_range(szx,my_comm),Range(0,szz-1));
   }

// Destructor
Cheb_2dmg::~Cheb_2dmg() {
   if (mg_precond) delete mg_precond;
   if (grad_2d) delete grad_2d;
   if (temp1) delete temp1;
   if (temp2) delete temp2;
   if (temp3) delete temp3;
   if (r2d) delete r2d;
   if (s2d) delete s2d;

   // Delete arrays used by GMRES as well
   for (vector<fbox *>::iterator i = r_free.begin(); i != r_free.end(); i++) {
      delete (*i)->gridbox;
      delete *i;
   }
   for (set<fbox *>::iterator i = r_used.begin(); i != r_used.end(); i++) {
      delete (*i)->gridbox;
      delete *i;
   }
   for (vector<ubox *>::iterator i = b_free.begin(); i != b_free.end(); i++) {
      delete (*i)->gridbox;
      delete *i;
   }
   for (set<ubox *>::iterator i = b_used.begin(); i != b_used.end(); i++) {
      delete (*i)->gridbox;
      delete *i;
   }
}

fbox *  Cheb_2dmg::alloc_resid() {
   // Allocate residual, using an already-allocated-and-freed one if possible
   fbox * r;
   if (r_free.size() == 0) { 
      // None are already allocated, so make a new one
      r = new fbox;
      r->gridbox = alloc_array(szx,1,szz,my_comm);
   } else {
      r = r_free.back();
      r_free.pop_back();
   } 
   r_used.insert(r);
   *r->gridbox = 0;
   r->mean = 0;
   return r;
}
void Cheb_2dmg::free_resid(fbox * & f) {
   // Free residual
   set<fbox *>::iterator loc = r_used.find(f);
   if (loc == r_used.end()) {
      fprintf(stderr,"ERROR: Freeing a GMRES residual that was not properly allocated\n");
      abort();
   } else {
      r_free.push_back(f);
      r_used.erase(loc);
   }
   f = 0;
   return;
}
ubox *  Cheb_2dmg::alloc_basis() {
   // Allocate basis, using an already-allocated-and-freed one if possible
   ubox * b;
   if (b_free.size() == 0) { 
      // None are already allocated, so make a new one
      b = new ubox;
      b->gridbox = alloc_array(szx,1,szz,my_comm);
   } else {
      b = b_free.back();
      b_free.pop_back();
   } 
   b_used.insert(b);
   *b->gridbox = 0;
   b->sigma = 0;
   return b;
}
void Cheb_2dmg::free_basis(ubox * & f) {
   // Free basis
   set<ubox *>::iterator loc = b_used.find(f);
   if (loc == b_used.end()) {
      fprintf(stderr,"ERROR: Freeing a GMRES basis that was not properly allocated\n");
      abort();
   } else {
      b_free.push_back(f);
      b_used.erase(loc);
   }
   f = 0;
   return;
}

double Cheb_2dmg::resid_dot(fbox * & l, fbox * & r) {
//   double s = pssum(sum(*l->gridbox * *r->gridbox),my_comm);
//   return pssum(sum(*l->gridbox * *r->gridbox),my_comm);
//   if (indefinite_problem)
//      s = s + l->mean*r->mean/(szx*szz);
   double s = 0;
   int imin, imax;
   if (type_x == CHEBY) {
      imin = max(1,l->gridbox->lbound(firstDim));
      imax = min(szx-2,l->gridbox->ubound(firstDim));
   } else {
      imin = l->gridbox->lbound(firstDim);
      imax = l->gridbox->ubound(firstDim);
   }
   // Interior points
   if (type_x == CHEBY) {
      for (int i = imin; i <= imax; i++) 
         for (int j = 1; j <= szz-2; j++)
            s = s + (*l->gridbox)(i,0,j)*(*r->gridbox)(i,0,j)/(szx*szz)*sin(M_PI*i/(szx-1))*sin(M_PI*j/(szz-1));
   } else {
      for (int i = imin; i <= imax; i++) 
         for (int j = 1; j <= szz-2; j++)
            s = s + (*l->gridbox)(i,0,j)*(*r->gridbox)(i,0,j)/(szx*szz)*sin(M_PI*j/(szz-1));
   }
   // Boudnary points
   for (int i = l->gridbox->lbound(firstDim); i <= l->gridbox->ubound(firstDim); i++) {
      s = s + (*l->gridbox)(i,0,0)*(*r->gridbox)(i,0,0)/szx/4;
   }
   for (int i = l->gridbox->lbound(firstDim); i <= l->gridbox->ubound(firstDim); i++) {
      s = s + (*l->gridbox)(i,0,szz-1)*(*r->gridbox)(i,0,szz-1)/szx/4;
   }
   if (type_x == CHEBY) {
      if (l->gridbox->lbound(firstDim)==0) {
         // have left boundary
         for (int j = 1; j < szz-1; j++)
            s = s + (*l->gridbox)(0,0,j)*(*r->gridbox)(0,0,j)/szz/4;
      }
      if (l->gridbox->ubound(firstDim)==szx-1) {
         // have right boundary
         for (int j = 1; j < szz-1; j++)
            s = s + (*l->gridbox)(szx-1,0,j)*(*r->gridbox)(szx-1,0,j)/szz/4;
      }
   }
   // indefinite problem
   if (indefinite_problem)
      s = s + l->mean*r->mean;
   return pssum(s,my_comm);
}

double Cheb_2dmg::basis_dot(ubox * & l, ubox * & r) {
   double s = pssum(sum(*l->gridbox * *r->gridbox),my_comm);
   if (indefinite_problem)
      s = s + l->sigma * r->sigma / (szx*szz);
   return s;
}

void Cheb_2dmg::resid_copy(fbox * & copyto, fbox * & copyfrom) {
   *copyto->gridbox = *copyfrom->gridbox;
   copyto->mean = copyfrom->mean;
}

void Cheb_2dmg::basis_copy(ubox * & copyto, ubox * & copyfrom) {
   *copyto->gridbox = *copyfrom->gridbox;
   copyto->sigma = copyfrom->sigma;
}

void Cheb_2dmg::resid_fma(fbox * & lhs, fbox * & rhs, double scale) {
   *lhs->gridbox = *lhs->gridbox + scale* *rhs->gridbox;
   lhs->mean = lhs->mean + scale*rhs->mean;
}

void Cheb_2dmg::basis_fma(ubox * & lhs, ubox * & rhs, double scale) {
   *lhs->gridbox = *lhs->gridbox + scale*(*rhs->gridbox);
   lhs->sigma = lhs->sigma + scale*rhs->sigma;
}

void Cheb_2dmg::resid_scale(fbox * & vec, double scale) {
   *vec->gridbox = scale* *vec->gridbox;
   vec->mean *= scale;
}

void Cheb_2dmg::basis_scale(ubox * & vec, double scale) {
   *vec->gridbox = *vec->gridbox*scale;
   vec->sigma *= scale;
}

// Functions special to Cheb_2dmg, rather than implementing generic behaviour:

void Cheb_2dmg::build_precond() {
   // Slightly misnamed, this function allocates grad_2d and mg_precond, with
   // proper parameters.  The first is the 2D gradient operator, which Cheb_2dmg
   // uses for computing the Laplacian and other bits, and the second is the
   // multigrid preconditioner used for the approximate inverse.

   // Temporary arrays
   DTArray & t1 = *temp1, & t2 = *temp2, & t3 = *temp3;
   DTArray * v_jac;

   blitz::firstIndex ii;

   // First up, create grad_2d
   // If we already have one, delete it.
   if (grad_2d) delete grad_2d;
   grad_2d = new Grad(szx,1,szz,type_x,NONE,CHEBY,my_comm);

   // Now, get the Jacobian from grad_3d, carve off a relevant 2D slice, and
   // assign it to the 2d gradient's jacobian.
   double c_jac; // entry for constant jacobian
   // xx
   grad_3d->get_jac(firstDim,firstDim,c_jac,v_jac);
   if (!v_jac) {// constant
      grad_2d->set_jac(firstDim,firstDim,c_jac,0);
   } else {
      t1 = (*v_jac)(Range::all(),Range(0,0),Range::all());
      grad_2d->set_jac(firstDim,firstDim,0,&t1);
   }
   // xz
   grad_3d->get_jac(firstDim,thirdDim,c_jac,v_jac);
   if (!v_jac) {// constant
      grad_2d->set_jac(firstDim,thirdDim,c_jac,0);
   } else {
      t1 = (*v_jac)(Range::all(),Range(0,0),Range::all());
      grad_2d->set_jac(firstDim,thirdDim,0,&t1);
   }
   // zx
   grad_3d->get_jac(thirdDim,firstDim,c_jac,v_jac);
   if (!v_jac) {// constant
      grad_2d->set_jac(thirdDim,firstDim,c_jac,0);
   } else {
      t1 = (*v_jac)(Range::all(),Range(0,0),Range::all());
      grad_2d->set_jac(thirdDim,firstDim,0,&t1);
   }
   grad_3d->get_jac(thirdDim,thirdDim,c_jac,v_jac);
   if (!v_jac) {// constant
      grad_2d->set_jac(thirdDim,thirdDim,c_jac,0);
   } else {
      t1 = (*v_jac)(Range::all(),Range(0,0),Range::all());
      grad_2d->set_jac(thirdDim,thirdDim,0,&t1);
   }

   // Now, create the multigrid solver
   if (mg_precond) delete mg_precond;
   if (type_x == CHEBY) {
      // No x symmetry, so the grids include boundaries
      Array<double,1> xvals(szx), zvals(szz);
      // The base grids for both are the standard Chebyshev grids
      xvals = cos(ii*M_PI/(szx-1));
      zvals = cos(ii*M_PI/(szz-1));
      mg_precond = new MG_Solver(xvals,zvals,SYM_NONE,my_comm);
   } else {
      // The problem includes symmetry, so the x grid does -not- have 
      // boundaries built-in.  Instead, the xgrid includes ghost points
      // just beyond the boundaries.
      Array<double,1> xvals(szx+2), zvals(szz);
      zvals = cos(ii*M_PI/(szz-1));
      xvals = (ii-0.5)*M_PI/szx;
      // If the domain is Fourier-periodic, then the base length is 2pi, otherwise 1pi
      if (type_x == FOURIER) {
         xvals = xvals*2;
         mg_precond = new MG_Solver(xvals,zvals,SYM_PERIODIC,my_comm);
      } else if (type_x == SINE) {
         mg_precond = new MG_Solver(xvals,zvals,SYM_ODD,my_comm);
      } else {
         assert(type_x == COSINE);
         mg_precond = new MG_Solver(xvals,zvals,SYM_EVEN,my_comm);
      }
   }

   // Now, with the solvers allocated, we need to define the Laplacian (_xx + _zz)
   // in terms of what it looks like on the computational box.

   // The basic form is:
   // [_x ; _z] = [J11 J12;J21 J22]*[_a; _b]
   // where _a and _b are on the computational box.  Doing the multiplication
   // gives [_x ; _z] = [J11*_a + J12*_b; J21*_a + J22*_b]
   // and dot-with-itself gives something ugly, made even worse because given
   // the form of grad_2d we will actually take the -x- and -z- derivatives of the
   // Jxy entries.  So, written out it looks something like:
   // [_xx + _zz] = 
   //  (J11^2+J21^2)*_aa + (J12*J11+J21*J22)*_ab + (J12^2+J22^2)*_bb +
   //  (J11_x+J21_z)*_a  + (J12_x+J22_z)*_b
   //
   // Simple, ain't it?

   // The definition is done all at once, so we need 5 temporary arrays.  We can
   // re-use t1, t2, and t3 from above, but that's not quite enough.  So, allocate
   // two residual vectors -- when we free them after this, we'll promptly re-use
   // them during GMRES proper.

   fbox * r1 = alloc_resid(), * r2 = alloc_resid();
   DTArray & t4 = *(r1->gridbox), & t5 = *(r2->gridbox);
   
   /* To take x-derivatives of Jacobian entries, we have to be careful
     with symmetry.  Odd symmetry on a Jacobian entry doesn't make much
    sense. */
   S_EXP sx_type=type_x;
   if (type_x == SINE) sx_type = COSINE;

   // The schematic is t1*_aa + t2*_bb + t3*_ab + t4*_a + t5*_b.

   // I lied about things being simple.  The MG_Solver wants to take -2D- arrays,
   // and grad_2d will work on (trivially) -3D- arrays.  So, we'll create 2D arrays
   // that are views of the above data.
//   TinyVector<int,3> lbound, extent;
//   lbound = t4.lbound();
//   extent = t4.extent();
   Array<double,2> t1_2d(t1(Range::all(),0,Range::all())),
                  t2_2d(t2(Range::all(),0,Range::all())),
                  t3_2d(t3(Range::all(),0,Range::all())),
                  t4_2d(t4(Range::all(),0,Range::all())),
                  t5_2d(t5(Range::all(),0,Range::all()));

   /* T1, T2, and T3 do not involve derivatives.  T4 and T5 do. */

   /* J11 */
   grad_2d->get_jac(firstDim,firstDim,c_jac,v_jac);
   if (!v_jac) { // constant
      t1 = c_jac;
      t4 = 0;
   } else {
      t1 = *v_jac;
      grad_2d->setup_array(v_jac,sx_type,NONE,CHEBY);
      grad_2d->get_dx(&t4);
   }
   // J22
   grad_2d->get_jac(thirdDim,thirdDim,c_jac,v_jac);
   if (!v_jac) { // constant
      t2 = c_jac;
      t5 = 0;
   } else {
      t2 = *v_jac;
      grad_2d->setup_array(v_jac,sx_type,NONE,CHEBY);
      grad_2d->get_dz(&t5);
   }
   // J12
   grad_2d->get_jac(firstDim,thirdDim,c_jac,v_jac);
   if (!v_jac) { // constant
      t3 = t1*c_jac;
   } else {
      t3 = t1*(*v_jac);
      grad_2d->setup_array(v_jac,sx_type,NONE,CHEBY);
      grad_2d->get_dx(&t5,true);
   }
   // J21
   grad_2d->get_jac(thirdDim,firstDim,c_jac,v_jac);
   if (!v_jac) { // constant
      t3 = t3 + c_jac*t2;
      t1 = t1*t1+c_jac*c_jac;
   } else {
      t3 = t3 + (*v_jac)*t2;
      t1 = t1*t1+(*v_jac)*(*v_jac);
      grad_2d->setup_array(v_jac,sx_type,NONE,CHEBY);
      grad_2d->get_dz(&t4,true);
   }
   // J12 again, to balance off t2
   grad_2d->get_jac(firstDim,thirdDim,c_jac,v_jac);
   if (!v_jac) { //constant
      t2 = t2*t2 + c_jac*c_jac;
   } else {
      t2 = t2*t2 + (*v_jac)*(*v_jac);
   }

   // Now, t1->t5 are set properly, so assign them to the MG operator
//   fprintf(stderr,"PROBLEM SETUP\n");
//   cerr << t1_2d << t2_2d << t3_2d << t4_2d << t5_2d;
   mg_precond->problem_setup(t1_2d,t2_2d,t3_2d,t4_2d,t5_2d); 


   //  Now, initialize the boundary dx/dz arrays for Neumann-type BCs.
   //  This involves a bit of math on the Jacobian, but it isn't -too- complicated.

   /* The largest trick is that the Jacobian we use for differentiation isn't
      really the Jacobian -- it's the transpose.  That little detail took an
      embarassingly long time to figure out.   The matrix multiplication for
      differentiation is:

      d(alpha_i)/d(x_j)*df/d(alpha_i) = df/d(x_j)

      where (alpha_i) is the computational coordinate.  This gives a (2D) matrix
      of [ a_x b_x; a_z b_z].  In reality, the Jacobian for vector transforms is
      the transpose:
      [a_x a_z; b_x b_z] * [x; y] = [a; b] (x,y,a,b being vectors here)

      At the top/bottom (left/right) boundaries, a (b) is the tangent vector.
      Expressing that in terms of x/y lets us easily find the normal (switch
      terms, flip a sign).  Then, we make sure it's the outwards normal:

      at a (b) = minimum (first index of the array), the normal is in the
      NEGATIVE b (a) direction.

      at a (b) = maximum (upper bound of the array), the normal is in the
      POSITIVE b (a) direction.

      */


   { // Top/bottom
      Range lxrange = split_range(szx,my_comm);
      Array<double,1> alpha_x(lxrange), alpha_z(lxrange),
                     beta_x(lxrange), beta_z(lxrange),
                     det(lxrange);

      {
         // Bottom -- normal in -b direction
         // Get Jacobian terms
         grad_2d->get_jac(firstDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_x = c_jac;
         else
            alpha_x = (*v_jac)(lxrange,0,0);
         grad_2d->get_jac(thirdDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_z = c_jac;
         else
            alpha_z = (*v_jac)(lxrange,0,0);
         grad_2d->get_jac(firstDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_x = c_jac;
         else
            beta_x = (*v_jac)(lxrange,0,0);
         grad_2d->get_jac(thirdDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_z = c_jac;
         else
            beta_z = (*v_jac)(lxrange,0,0);

         det = alpha_x*beta_z-beta_x*alpha_z;

         // The tangent vector (alpha) is given by [bz/D; -bx/D]
         // so flip for the candidate normal.  Also nomalize while
         // we have this around.

         dx_bot = beta_x/sqrt(beta_x*beta_x + beta_z*beta_z);
         dz_bot = beta_z/sqrt(beta_x*beta_x + beta_z*beta_z);

         /* At the bottom (beta=max), we want the normal to be in the
            outward beta direction.  Beta is [-az/D; ax/D] */

         if (any(dx_bot*(-alpha_z/det) + dz_bot*alpha_x/det < 0)) {
            // If any d/dnormal*d/dt > 0, it better all be or we've screwed up
            // orientations somehow
            // This -should be- a parallel check, but first of all it is unlikely
            // that the orientation would flip exactly at processor boundaries,
            // and secondly if the orientation flips on one but not all processors
            // the if statement above won't necessarily trigger, making a parallel
            // check here into a deadlock.
            assert(all(dx_bot*(-beta_x/det)+dz_bot*(alpha_x/det) < 0));
            dx_bot = -dx_bot;
            dz_bot = -dz_bot;
         }

         // Now, repeat for the top
         // Get Jacobian terms
         grad_2d->get_jac(firstDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_x = c_jac;
         else
            alpha_x = (*v_jac)(lxrange,0,szz-1);
         grad_2d->get_jac(thirdDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_z = c_jac;
         else
            alpha_z = (*v_jac)(lxrange,0,szz-1);
         grad_2d->get_jac(firstDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_x = c_jac;
         else
            beta_x = (*v_jac)(lxrange,0,szz-1);
         grad_2d->get_jac(thirdDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_z = c_jac;
         else
            beta_z = (*v_jac)(lxrange,0,szz-1);

         det = alpha_x*beta_z-beta_x*alpha_z;

         // The tangent vector (alpha) is given by [bz/D; -bx/D]
         // so flip for the candidate normal.  Also nomalize while
         // we have this around.

         dx_top = beta_x/sqrt(beta_x*beta_x + beta_z*beta_z);
         dz_top = beta_z/sqrt(beta_x*beta_x + beta_z*beta_z);

         /* At the top (beta=0), we want the normal to be in the
            negative beta direction.  Beta is [-az/D; ax/D] */

         if (any(dx_top*(-alpha_z/det) + dz_top*alpha_x/det > 0)) {
            // If any d/dnormal*d/dt > 0, it better all be or we've screwed up
            // orientations somehow
            assert(all(dx_top*(-alpha_z/det)+dz_top*(alpha_x/det) > 0));
            dx_top = -dx_top;
            dz_top = -dz_top;
         }




      }
   }
   { // Left/right
      Array<double,1> alpha_x(szz), alpha_z(szz), beta_x(szz), beta_z(szz), det(szz);
      Range rall = Range::all();

      if (t1.lbound(firstDim) == 0) {
         // Left
         // Get Jacobian terms
         grad_2d->get_jac(firstDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_x = c_jac;
         else
            alpha_x = (*v_jac)(0,0,rall);
         grad_2d->get_jac(thirdDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_z = c_jac;
         else
            alpha_z = (*v_jac)(0,0,rall);
         grad_2d->get_jac(firstDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_x = c_jac;
         else
            beta_x = (*v_jac)(0,0,rall);
         grad_2d->get_jac(thirdDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_z = c_jac;
         else
            beta_z = (*v_jac)(0,0,rall);

         det = alpha_x*beta_z-beta_x*alpha_z;

         // Beta is now the tangent vector, and in x/z space that's
         // proportional to [-az;ax].  Flip for normal, and we have
         // [ax; +az].  Also normalize

         dx_left = alpha_x/sqrt(alpha_x*alpha_x+alpha_z*alpha_z);
         dz_left = alpha_z/sqrt(alpha_x*alpha_x+alpha_z*alpha_z);
 
         // The left boundary corresponds to alpha being its maximum
         // (Chebyshev grids remain weird), so we want normal dot a > 0
         // and a = [bz/det; -bx/det]

         if (any(dx_left*(beta_z/det) + dz_left*(-beta_x/det) < 0)) {
            // If any d/dnormal*d/dt > 0, it better all be or we've screwed up
            // orientations somehow
            assert(all((dx_left*(beta_z/det) + dz_left*(-beta_x/det)) < 0));
            dx_left = -dx_left;
            dz_left = -dz_left;
         }
      }
      if (t1.ubound(firstDim) == szx-1) {
//         fprintf(stderr,"Making right\n");
         // Right
         // Get Jacobian terms
         grad_2d->get_jac(firstDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_x = c_jac;
         else
            alpha_x = (*v_jac)(szx-1,0,rall);
         grad_2d->get_jac(thirdDim,firstDim,c_jac,v_jac);
         if (!v_jac) 
            alpha_z = c_jac;
         else
            alpha_z = (*v_jac)(szx-1,0,rall);
         grad_2d->get_jac(firstDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_x = c_jac;
         else
            beta_x = (*v_jac)(szx-1,0,rall);
         grad_2d->get_jac(thirdDim,thirdDim,c_jac,v_jac);
         if (!v_jac) 
            beta_z = c_jac;
         else
            beta_z = (*v_jac)(szx-1,0,rall);

         det = alpha_x*beta_z-beta_x*alpha_z;
         // Beta is now the tangent vector, and in x/z space that's
         // proportional to [-az;ax].  Flip for normal, and we have
         // [ax; +az].  Also normalize

         dx_right = alpha_x/sqrt(alpha_x*alpha_x+alpha_z*alpha_z);
         dz_right = alpha_z/sqrt(alpha_x*alpha_x+alpha_z*alpha_z);
 
         // The left boundary corresponds to alpha being its maximum
         // (Chebyshev grids remain weird), so we want normal dot a > 0
         // and a = [bz/det; -bx/det]


         if (any(dx_right*(beta_z/det) + dz_right*(-beta_x/det) > 0)) {
            // If any d/dnormal*d/dt > 0, it better rall be or we've screwed up
            // orientations somehow
            assert(all(dx_right*(beta_z/det) + dz_right*(-beta_x/det) > 0));
            dx_right = -dx_right;
            dz_right = -dz_right;
         }
      }

   }

   /*
   fprintf(stderr,"left\n");
   cerr << dx_left << dz_left;
   fprintf(stderr,"right\n");
   cerr << dx_right << dz_right;
   fprintf(stderr,"bottom\n");
   cerr << dx_bot << dz_bot;
   fprintf(stderr,"top\n");
   cerr << dx_top << dz_top;

   MPI_Finalize(); exit(1);*/
      
   // Free the temporary resid vectors
   free_resid(r1);
   free_resid(r2);
}

void Cheb_2dmg::set_grad(Grad * in_grad) {
   if (in_grad != grad_3d) {
      grad_3d = in_grad;
      type_x = grad_3d->get_type(firstDim);
      build_precond();
   }
}

void Cheb_2dmg::set_bc(double h, double zdir, double zneum, S_EXP sym_type, double xdir, double xneum) {
   // Set the boundary conditions used in the solve
   if (type_x == sym_type && helm == h && dir_zbc == zdir && neum_zbc == zneum &&
         dir_xbc == xdir && neum_xbc == xneum)  {
      return;
   }
   type_x = sym_type;
   helm = h;
   dir_zbc = zdir;
   neum_zbc = zneum;
   neum_xbc = xneum;
   dir_xbc = xdir;
   if (h == 0 && dir_xbc == 0 && dir_zbc == 0 && type_x != SINE) {
      indefinite_problem = true;
   }

   // propagate the changes to the multigrid preconditioner.
   // ignore the neumann BC here for simplicity, FIXME.
   // This will involve some Jacobian math.

   if (type_x == SINE)
      mg_precond->set_x_symmetry(SYM_ODD);
   if (type_x == COSINE)
      mg_precond->set_x_symmetry(SYM_EVEN);

   mg_precond->helmholtz_setup(h);
   
   // Allocate arrays for BCs

   blitz::Array<double,1> dir_x(szz), da_left(szz), da_right(szz),
      db_left(szz), db_right(szz);
   Range lxrange = split_range(szx,my_comm);
   blitz::Array<double,1> dir_z(lxrange),
      db_bot(lxrange), db_top(lxrange),
      da_bot(lxrange), da_top(lxrange);

   // Dirichlet conditions
   dir_x = xdir; dir_z = zdir;


   // Since the preconditioner works on a computational box,
   // we have to give it Neumann BCs on a rectangular grid.  That
   // means taking our physical-space dx/dz (computed when we
   // got the gradient operator) and applying the Jacobian to turn
   // them into da/db.

   // Doubles for possibly constant Jacobian terms
   double c_ax, c_bx, c_az, c_bz;
   // DTArray * for the variable terms
   DTArray * v_ax, * v_bx, * v_az, * v_bz;
   grad_2d->get_jac(firstDim,firstDim,c_ax,v_ax);
   grad_2d->get_jac(firstDim,thirdDim,c_bx,v_bx);
   grad_2d->get_jac(thirdDim,firstDim,c_az,v_az);
   grad_2d->get_jac(thirdDim,thirdDim,c_bz,v_bz);

   /* Print out dx/dz for sides */
/*   fprintf(stderr,"Physical Neumann coefs\n");
   fprintf(stderr,"Left\n");
   cerr << dx_left << dz_left;
   fprintf(stderr,"Right\n");
   cerr << dx_right << dz_right;
   fprintf(stderr,"Bottom\n");
   cerr << dx_bot << dz_bot;
   fprintf(stderr,"Top\n");
   cerr << dx_top << dz_top;*/

   // Top & Bottom
   if (!v_ax) { // constant ax jacobian
      da_bot = c_ax*dx_bot;
      da_top = c_ax*dx_top;
   } else {
      da_bot = dx_bot*(*v_ax)(lxrange,0,0);
      da_top = dx_top*(*v_ax)(lxrange,0,szz-1);
   }
   if (!v_bx) { // constant bx jac
      db_bot = c_bx*dx_bot;
      db_top = c_bx*dx_top;
   } else {
      db_bot = dx_bot*(*v_bx)(lxrange,0,0);
      db_top = dx_top*(*v_bx)(lxrange,0,szz-1);
   }
   if (!v_az) {
      da_bot += c_az*dz_bot;
      da_top += c_az*dz_top;
   } else {
      da_bot += dz_bot*(*v_az)(lxrange,0,0);
      da_top += dz_top*(*v_az)(lxrange,0,szz-1);
   }
   if (!v_bz) {
      db_bot += c_bz*dz_bot;
      db_top += c_bz*dz_top;
   } else {
      db_bot += dz_bot*(*v_bz)(lxrange,0,0);
      db_top += dz_top*(*v_bz)(lxrange,0,szz-1);
   }

   if (type_x == CHEBY) {
      Range all = Range::all();
      // Left
      if (temp1->lbound(firstDim) == 0) {
         if (!v_ax) da_left = c_ax*dx_left;
         else da_left = dx_left*(*v_ax)(0,0,all);
         if (!v_bx) db_left = c_bx*dx_left;
         else db_left = dx_left*(*v_bx)(0,0,all);
         if (!v_az) da_left += c_az*dz_left;
         else da_left += dz_left*(*v_az)(0,0,all);
         if (!v_bz) db_left += c_bz*dz_left;
         else db_left += dz_left*(*v_bz)(0,0,all);
      }
      // BCs for the multigrid method must be initialized everywhere,
      // so copy the calculated da/db to all processors
      MPI_Bcast(da_left.data(),szz,MPI_DOUBLE,0,my_comm);
      MPI_Bcast(db_left.data(),szz,MPI_DOUBLE,0,my_comm);
   
      int numproc; MPI_Comm_size(my_comm,&numproc);
      // Right
      if (temp1->ubound(firstDim) == szx-1) {
         if (!v_ax) da_right = c_ax*dx_right;
         else da_right = dx_right*(*v_ax)(szx-1,0,all);
         if (!v_bx) db_right = c_bx*dx_right;
         else db_right = dx_right*(*v_bx)(szx-1,0,all);
         if (!v_az) da_right += c_az*dz_right;
         else da_right += dz_right*(*v_az)(szx-1,0,all);
         if (!v_bz) db_right += c_bz*dz_right;
         else db_right += dz_right*(*v_bz)(szx-1,0,all);
      }
      // BCs for the multigrid method must be initialized everywhere,
      // so copy the calculated da/db to all processors
      MPI_Bcast(da_right.data(),szz,MPI_DOUBLE,numproc-1,my_comm);
      MPI_Bcast(db_right.data(),szz,MPI_DOUBLE,numproc-1,my_comm);
   }
   // Apply the Neumann condition to the calculated vectors now
   da_left *= xneum;
   db_left *= xneum;
   da_right *= xneum;
   db_right *= xneum;
   da_top *= zneum;
   db_top *= zneum;
   da_bot *= zneum;
   db_bot *= zneum;
  /* 
   fprintf(stderr,"left\n");
   cerr << da_left << db_left;
   fprintf(stderr,"right\n");
   cerr << da_right << db_right; 
   fprintf(stderr,"bottom\n");
   cerr << da_bot << db_bot;
   fprintf(stderr,"top\n");
   cerr << da_top << db_top; 
   */


   mg_precond->bc_setup(firstDim,dir_x,db_left,da_left,
                                 dir_x,db_right,da_right);
   mg_precond->bc_setup(secondDim,dir_z,db_bot,da_bot,
                                 dir_z,db_top,da_top);
}
   

void Cheb_2dmg::precondition(fbox * & rhs, ubox * & soln) {
   // The fbox and ubox pointers contain 3D arrays and mg_precond
   // wants 2D arrays, so create references.
   Range all = Range::all();
//   Array<double,2> r2d((*rhs->gridbox)(all,0,all)), s2d((*soln->gridbox)(all,0,all));
   // The multigrid preconditioner works with standard, c-ordered arrays.   If this
   // is a multiprocessor case, this is not what we have.
   *r2d = (*rhs->gridbox)(all,0,all);
   blitz::secondIndex jj;
   blitz::firstIndex ii;
//   Array<double,2> t1(r2d->copy()), t2(r2d->copy());
//   Array<double,2> t1(r2d->lbound(),r2d->extent()),
//                   t2(r2d->lbound(),r2d->extent());
//   SYNC(cerr << *r2d;cerr.flush());
//   fprintf(stderr,"Mean condition: %g\n",rhs->mean);
   //fprintf(stderr,"Precond: max rhs %g\n",max(abs(*(rhs->gridbox))));
   mg_precond->cycle(CYCLE_F,*r2d,*s2d, rhs->mean,soln->sigma,0,1,2);
   //mg_precond->cycle(CYCLE_V,*r2d,*s2d, rhs->mean,soln->sigma,1,0,0);
   //fprintf(stderr,"Precond: max soln %g\n",max(abs(*s2d)));

//   mg_precond->cycle(CYCLE_F,*r2d,*s2d, rhs->mean,soln->sigma,0,0,0);
//   *s2d = cos(ii*M_PI/(szx-1))-(0.2*(cos(jj*M_PI/(szz-1)) - 0.2*cos(ii*M_PI/(szz-1))));
//   *s2d = cos(ii*M_PI/(szx-1));
   (*soln->gridbox)(all,0,all) = *s2d;
//   SYNC(cerr << *s2d; cerr.flush());
//   MPI_Finalize(); exit(1);
//   SYNC(cerr << *s2d; cerr.flush());
//   fprintf(stderr,"Returned sigma: %g\n",soln->sigma);
//   MPI_Finalize(); exit(1);
//   mg_precond->apply_operator(*s2d,t1);
//   SYNC(cerr << t1; cerr.flush());
//   t2 = *r2d-(t1+soln->sigma);
//   SYNC(cerr << t2; cerr.flush());
//   double mfd = psmax(max(abs(t2)),my_comm);
//   double nfd = sqrt(pssum(sum(t2*t2),my_comm))/szx/szz;
//   if (master(my_comm))
//      fprintf(stderr,"Max FD norm %g inf %g 2\n",mfd,nfd);;
//   MPI_Finalize(); exit(1);
//   mg_precond->cycle(CYCLE_V,t2,*s2d,rhs->mean,soln->sigma,1,0,0);
//   SYNC(cerr << *s2d;cerr.flush());
//   MPI_Finalize();exit(1);
//   (*soln->gridbox)(all,0,all) = (*soln->gridbox)(all,0,all) + *s2d;
//   fprintf(stderr,"Preconditioning\n");
//   cerr << r2d << s2d;
   // Now, if there's a Nyquist component to the solution on a non-Chebyshev
   // grid, we have to remove it.  On a sine or Fourier grid, the derivative
   // (spectrally) of a wave at the Nyquist frequency is 0 (introducing a null
   // space to the Laplacian operator), but the FD derivative does not do
   // the same thing.  Thus, we need to filter out any Nyquist frequency from
   // the preconditioned guess.
//   fprintf(stderr,"%d %d %d %d %d\n",type_x,SINE,FOURIER,szx,szx%2);
   if (0 && (type_x == SINE || (type_x == FOURIER && (szx % 2) == 0))) {
      Trans1D * gt = grad_2d->get_trans(firstDim);
      gt->forward_transform(soln->gridbox,type_x);
      if (type_x == SINE) {
         if (gt->get_real_temp()->ubound(firstDim) == szx-1) {
            int lb = max(1,gt->get_real_temp()->lbound(thirdDim));
            int ub = min(szz-2,gt->get_real_temp()->ubound(thirdDim));
            (*gt->get_real_temp())(szx-1,0,Range(lb,ub)) = 0;
         }
         (*gt->get_real_temp()) /= gt->norm_factor();
      } else if (type_x == FOURIER) {
//         fprintf(stderr,"%d %d\n",gt->get_complex_temp()->ubound(firstDim),szx/2);
//         cerr << gt->wavenums() << endl;

//   MPI_Finalize(); exit(1);
         if (gt->get_complex_temp()->ubound(firstDim) == (szx/2)) {
//            fprintf(stderr,"Removing Nyq\n");
            int lb = max(1,gt->get_complex_temp()->lbound(thirdDim));
            int ub = min(szz-2,gt->get_complex_temp()->ubound(thirdDim));
            (*gt->get_complex_temp())(szx/2,0,Range(lb,ub)) = 0;
         }
         (*gt->get_complex_temp()) /= gt->norm_factor();
      }
//   MPI_Finalize(); exit(1);
      gt->back_transform(soln->gridbox,type_x);
//      double nyqfreq;
//      blitz::firstIndex ii;
//      nyqfreq = pssum(sum((*soln->gridbox)(Range(1,szz-1),0,all)*(2*(ii%2)-1)))/(szx*(szz-2));
//      (*soln->gridbox)(Range(1,szz-1),0,all) =
//         (*soln->gridbox)(Range(1,szz-1),0,all) - nyqfreq*(2*(ii%2)-1);
//      if (max(*(soln->gridbox)) > 100) {
//         write_array(*(soln->gridbox),"precond_dump");
//         write_reader(*(soln->gridbox),"precond_dump");
//         MPI_Finalize();
//         exit(1);
//      }
   }
//      write_array(*(soln->gridbox),"precond_dump");
//      write_reader(*(soln->gridbox),"precond_dump");
   
   
}

void Cheb_2dmg::matrix_multiply(ubox * & lhs, fbox * & rhs) {
   // Applies Laplacian_u  + helm*u
   // Assign key temporaries to humane labels
//   Range all = Range::all();
//   Array<double,2> lhs_2d((*lhs->gridbox)(all,0,all)), rhs_2d((*rhs->gridbox)(all,0,all));
//   mg_precond->apply_operator(lhs_2d,rhs_2d);
//   rhs->mean = 0;
//   fprintf(stderr,"Matrix multiply\n");
//   cerr << lhs_2d << rhs_2d;
//   return;
   DTArray & dx = *temp1, & dz = *temp2;

   Range all = Range::all();

//   fprintf(stderr,"Matrix multiply\n");
//   fprintf(stderr,"matrix_multiply lhs max %g\n",max(abs(*(lhs->gridbox))));
//   cerr << *lhs->gridbox;
//   fprintf(stderr,"Sigma: %g\n",lhs->sigma);

   // Compute grad(lhs)
//   SYNC(cerr << *lhs->gridbox;cerr.flush());
   grad_2d->setup_array(lhs->gridbox,type_x,NONE,CHEBY);
   grad_2d->get_dx(&dx);
   grad_2d->get_dz(&dz);

//   fprintf(stderr,"Dx\n");
//   fprintf(stderr,"Dz\n");

   // lhs_xx
   if (type_x == SINE)
      grad_2d->setup_array(&dx,COSINE,NONE,CHEBY);
   else if (type_x == COSINE)
      grad_2d->setup_array(&dx,SINE,NONE,CHEBY);
   else
      grad_2d->setup_array(&dx,type_x,NONE,CHEBY);
      
   grad_2d->get_dx(rhs->gridbox);
   // lhs_zz
   if (type_x == SINE)
      grad_2d->setup_array(&dz,COSINE,NONE,CHEBY);
   else if (type_x == COSINE)
      grad_2d->setup_array(&dz,SINE,NONE,CHEBY);
   else
      grad_2d->setup_array(&dz,type_x,NONE,CHEBY);
   grad_2d->get_dz(rhs->gridbox,true);

   // apply helmholtz

   *rhs->gridbox = *rhs->gridbox + helm*(*lhs->gridbox);

   // apply BCs.  Only Dirichlet for now.
//   if (type_x == CHEBY) {
//      if (rhs->gridbox->lbound(firstDim) == 0)
//         (*rhs->gridbox)(0,0,all) =
//      if (rhs->gridbox->ubound(firstDim) == szx-1)
//         (*rhs->gridbox)(szx-1,0,all) =
//   }
//
//   (*rhs->gridbox)(all,0,0) =
//   (*rhs->gridbox)(all,0,szz-1) =
//
    {
      // Apply Neumann BCs
      if (type_x == CHEBY) {
         // Left and right
         if (rhs->gridbox->lbound(firstDim) == 0) {
            (*rhs->gridbox)(0,0,all) = dir_xbc*(*lhs->gridbox)(0,0,all) +
               neum_xbc*(dx_left*dx(0,0,all)+dz_left*dz(0,0,all));
         }
         if (rhs->gridbox->ubound(firstDim) == szx-1) {
            (*rhs->gridbox)(szx-1,0,all) = dir_xbc*(*lhs->gridbox)(szx-1,0,all) +
               neum_xbc*(dx_right*dx(szx-1,0,all) + dz_right*dz(szx-1,0,all));
         }
//   SYNC(cerr << *rhs->gridbox;cerr.flush());
      }
//      cerr << dx_bot << dz_bot;
      // Top and bottom
      (*rhs->gridbox)(all,0,0) = dir_zbc*(*lhs->gridbox)(all,0,0) +
         neum_zbc*(dx_bot*dx(all,0,0)+dz_bot*dz(all,0,0));
      (*rhs->gridbox)(all,0,szz-1) = dir_zbc*(*lhs->gridbox)(all,0,szz-1) +
         neum_zbc*(dx_top*dx(all,0,szz-1)+dz_top*dz(all,0,szz-1));
//   SYNC(cerr << *rhs->gridbox;cerr.flush());
   }
   if (indefinite_problem) {
      rhs->mean = pvsum(*lhs->gridbox,my_comm)/(szx*szz);
      (*rhs->gridbox) += lhs->sigma;
   }
//   SYNC(cerr << *rhs->gridbox;cerr.flush());
//   fprintf(stderr,"mean: %g\n",rhs->mean);
//   MPI_Finalize(); exit(1);
//   fprintf(stderr,"matrix_multiply rhs max %g\n",max(abs(*(rhs->gridbox))));
}

bool Cheb_2dmg::noisy() const {
   return false && master(my_comm);
}
   


/* Public interface of the 2D Poisson solver, simplified to assume uniform
   boundary conditions at all 4 (2 in a symmetric case) walls */
int poisson_2d(TArrayn::DTArray & resid, TArrayn::DTArray & soln, 
      TArrayn::Grad * gradient, Transformer::S_EXP type_x,
      double helmholtz, double zbc_a, double zbc_b,
      double xbc_a, double xbc_b) {

   static Cheb_2dmg * solver = 0;
   static GMRES_Solver<Cheb_2dmg> * gmresser = 0;
   static int cszx = 0; static int cszz = 0;
   static MPI_Comm comm = MPI_COMM_WORLD;
   static double warn_level = 1e-5;
   //int inn_its = 20, out_its = 2;
   int inn_its = 10, out_its = 3;

   if (xbc_a == 0 && xbc_b == 0) {
      xbc_a = zbc_a;
      xbc_b = zbc_b;
   }

   if (psall(all(resid==0),gradient->get_communicator())) {
      soln = 0;
      return 1;
   }

//   cerr << resid;

   static fbox * rhs = 0;
   static ubox * lhs = 0;
   if (!rhs) rhs = new fbox;
   if (!lhs) lhs = new ubox;
   rhs->gridbox = &resid;
   rhs->mean = 0;
   lhs->gridbox = &soln;
   lhs->sigma = 0;

   int szx, szy, szz;
   gradient->get_sizes(szx,szy,szz);
   if (szx != cszx || szz != cszz || !solver || 
         !gmresser || comm != gradient->get_communicator()) {
      if (solver) delete solver; 
      if (gmresser) delete gmresser;
      cszx = szx; cszz = szz;
      comm = gradient->get_communicator();
      solver = new Cheb_2dmg(cszx,cszz,comm);
      gmresser = new GMRES_Solver<Cheb_2dmg>(solver);
   }
   solver->set_grad(gradient);
   solver->set_bc(-helmholtz,zbc_a,zbc_b,type_x,xbc_a,xbc_b);

   double final_error;
   if (solver->noisy()) {
      fprintf(stderr,"Solving 2DMG problem (%g:%f-%f+%f-%f) with GMRES(%d,%d)\n",helmholtz,zbc_a,zbc_b,xbc_a,xbc_b,inn_its,out_its);
   }
   int itc = gmresser->Solve(lhs,rhs,1e-6,inn_its,out_its,&final_error);
   if (solver->noisy()) {
      fprintf(stderr,"Solve: %d its, %g error, %g sigma\n",itc,final_error,lhs->sigma);
   }
   if (itc < 0 && final_error > warn_level) {
      if (master()) {
         fprintf(stderr,"WARNING: GMRES(%d,%d) only solved the problem to %g residual level\n",inn_its,out_its,final_error);
         fprintf(stderr,"The problem was: H=%g, BCs(%g,%g)+(%g,%g)\n",helmholtz,zbc_a,zbc_b,xbc_a,xbc_b);
      }
      warn_level = final_error;
//      abort();
   }
   return(itc);
}

      

