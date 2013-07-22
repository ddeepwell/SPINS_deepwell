#include "TArray.hpp"
#include "T_util.hpp"
#include "gmres.hpp"
#include "Parformer.hpp"
#include <blitz/array.h>
#include <vector>
#include <set>
#include <stdio.h>
#include <iostream>

using TArrayn::DTArray;
using Transformer::Trans1D;
using TArrayn::deriv_cheb;
using TArrayn::firstDim; using TArrayn::secondDim; using TArrayn::thirdDim;

using std::vector;
using std::set;

extern "C" {
      extern void dgbsv_(int *, int *, int *, int *, double *,
                           int *, int *, double *, int *, int *);
      extern void dgbtrf_(int * m, int * n, int * kl, int * ku,
                           double * ab, int * ldab, int  * ipiv, 
                           int * info);
      extern void dgbtrs_(const char * trans, int * n, int * kl, int * ku,
                           int * nrhs, double * ab, int * ldab,
                           int * ipiv, double * b, int * ldb, 
                           int * info);
}

struct gridline {
      /* Struct to hold together the main vector itself (zline, after the typical
               dimension this will be used with) and an extra double for normalization
                     with Neumann-Neumann BCs */
      DTArray * __restrict__ zline;
         double extra;
      void print() {
//         std::cout << *zline;
//         printf("[+] %g [+]\n",extra);
      }
};
class cheb_d2 : public GMRES_Interface<gridline *, gridline *> {
   private:
      int size;
      DTArray * __restrict__ tline;
      Trans1D * __restrict__ chebformer;
      blitz::Array<double,2> band_d2;
      blitz::Array<int,1> pivot_factors;
      
      double factored; // Whether or not the preconditioner is already factorized

      /* BC values */
      double a_top, a_bot, b_top, b_bot;
      /* Helmholtz parameter */
      double helm;

      /* Length (eventually extend to Jacobian) */
      double length;

      void build_precond();
   public:
      cheb_d2(int sz, double length);
      void set_bc(double helmholtz, double atop, 
            double abot, double btop, double bbot);
      ~cheb_d2();
      RType alloc_resid();
      BType alloc_basis();
      void free_resid(RType &);
      void free_basis(BType &);
      void matrix_multiply(BType &, RType &);
      void precondition(RType &, BType &);
      double resid_dot(RType &, RType &);
      double basis_dot(BType &, BType &);
      void resid_copy(RType &, RType &);
      void basis_copy(RType &, RType &);
      void resid_fma(RType &, RType &, double);
      void basis_fma(BType &, BType &, double);
      void resid_scale(RType &, double);
      void basis_scale(RType &, double);
      bool noisy() const;

   private:
      vector<RType> r_free;
      vector<BType> b_free;
      set<RType> r_used;
      set<BType> b_used;
};
