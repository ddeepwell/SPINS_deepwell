/* Implementation of the gradient wrapper class */

#include "grad.hpp"
#include "Par_util.hpp"
#include "T_util.hpp"
#include <stdio.h>

using namespace Transformer;
namespace TArrayn {

   Grad::Grad(int szx, int szy, int szz, S_EXP tx, S_EXP ty, S_EXP tz,
         MPI_Comm c) {
      /* Copy over communicator, base size/type info */
      size_x = szx; size_y = szy; size_z = szz;
      type_x = tx; type_y = ty; type_z = tz;
      comm = c;

      /* Zero the jacobian, until it's set via the set_ method */
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            c_jac[i][j] = 0;
            v_jac[i][j] = 0;
         }
      }

      /* Get the proper array allocation semantics */
      a_lbound = alloc_lbound(size_x, size_y, size_z, comm);
      a_extent = alloc_extent(size_x, size_y, size_z, comm);
      a_storage = alloc_storage(size_x, size_y, size_z, comm);

      dda = ddb = ddc = 0;
      /* Allocate temporary arrays */
      if (size_x > 1) dda = new Array<double,3> (a_lbound,a_extent,a_storage);
      if (size_y > 1) ddb = new Array<double,3> (a_lbound,a_extent,a_storage);
      if (size_z > 1) ddc = new Array<double,3> (a_lbound,a_extent,a_storage);

      /* Create transform wrappers as necessary */
      T_a = T_b = T_c = 0;
      if (size_x > 1) 
         T_a = new Trans1D(size_x,size_y,size_z,firstDim,type_x,comm);
      if (size_y > 1)
         T_b = new Trans1D(size_x,size_y,size_z,secondDim,type_y,comm);
      if (size_z > 1)
         T_c = new Trans1D(size_x,size_y,size_z,thirdDim,type_z,comm);

   }

   Grad::~Grad() {
      /* Delete temporary arrays */
      if (dda) delete dda;
      if (ddb) delete ddb;
      if (ddc) delete ddc;
      if (T_a) delete T_a;
      if (T_b) delete T_b; 
      if (T_c) delete T_c;

      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            if (v_jac[i][j])
               delete v_jac[i][j];
         }
      }
   }

   void Grad::set_jac(Dimension d1, Dimension d2, double constant,
         blitz::Array<double,3> * varying) {
      if (v_jac[d1][d2]) delete v_jac[d1][d2];
      if (varying) {
         v_jac[d1][d2] = new DTArray(varying->copy());
         c_jac[d1][d2] = 0;
      } else  {
         c_jac[d1][d2] = constant;
         v_jac[d1][d2] = 0;
      }
   }

   void Grad::get_jac(Dimension d1, Dimension d2, double & constant,
         DTArray * & varying) const {
      if (v_jac[d1][d2]) {
         varying = v_jac[d1][d2];
         constant = 0;
      } else {
         varying = 0;
         constant = c_jac[d1][d2];
      }
   }

   bool Grad::constant_jac() const {
      return (!v_jac[0][0] && !v_jac[0][1] && !v_jac[0][2] &&
              !v_jac[1][0] && !v_jac[1][1] && !v_jac[1][2] &&
              !v_jac[2][0] && !v_jac[2][1] && !v_jac[2][2]);
   }

   bool Grad::constant_diagonal_jac() const {
      return (constant_jac() && diagonal_jac());
   }

   bool Grad::diagonal_jac() const {
      return (!v_jac[0][1] && !c_jac[0][1] && !v_jac[0][2] && !c_jac[0][2] &&
              !v_jac[1][0] && !c_jac[1][0] && !v_jac[1][2] && !c_jac[1][2] &&
              !v_jac[2][0] && !c_jac[2][0] && !v_jac[2][1] && !c_jac[2][1]);
   }

   bool Grad::constant_jac(Dimension d1, Dimension d2) const {
      return !v_jac[d1][d2];
   }

   bool Grad::zero_jac(Dimension d1, Dimension d2) const {
      return (!v_jac[d1][d2] && (c_jac[d1][d2] == 0));
   }

   void Grad::get_sizes(int & szx, int & szy, int & szz) const {
      szx = size_x; szy = size_y; szz = size_z;
      return;
   }

   MPI_Comm Grad::get_communicator() const {
      return comm;
   }

   /* Now, for the real meat of the problem */
   void Grad::setup_array(DTArray * ar, S_EXP t1, S_EXP t2, S_EXP t3) {

      /* Keep a record of this array for use in dx/dy/dz */
      my_array = ar;

      
      /* Since we're getting a new array to deal with, we don't possibly
         have any of the derivatives already */
      b_da = b_db = b_dc = false;

      type_x = t1; type_y = t2; type_z = t3;
   }

   /* Private function to save boilerplate of calculating d/d[abc] with the
      right kind of transform */
   void Grad::calculate_d(S_EXP type, Trans1D * tform, blitz::Array<double,3> * restrict dest) {
      switch(type) {
         case COSINE:
            deriv_dct(*my_array,*tform,*dest);
            break;
         case SINE:
            deriv_dst(*my_array,*tform,*dest);
            break;
         case FOURIER:
            deriv_fft(*my_array,*tform,*dest);
            break;
         case CHEBY:
            deriv_cheb(*my_array,*tform,*dest);
            break;
         default:
            fprintf(stderr,"Invalid expansion type (%d) in grad!\n",type);
            abort();
      }
   }

   void Grad::do_diff(DTArray * restrict dx_out, Dimension inDim) {
      if ((size_x > 1) && !zero_jac(inDim,firstDim)) {
         /* Need d/da */
         if (!b_da) { // need to calculate it
            calculate_d(type_x,T_a,dda);
            b_da = true;
         }
         if (constant_jac(inDim,firstDim)) {
            *dx_out = *dx_out + c_jac[inDim][firstDim]*(*dda);
         } else { // Variable jacobian
            *dx_out = *dx_out + (*v_jac[inDim][firstDim])*(*dda);
         }
      }

      if ((size_y > 1) && !zero_jac(inDim,secondDim)) {
         /* Need d/db */
         if (!b_db) {
            calculate_d(type_y,T_b,ddb);
            b_db = true;
         }
         if (constant_jac(inDim,secondDim)) {
            *dx_out = *dx_out + c_jac[inDim][secondDim]*(*ddb);
         } else {
            *dx_out = *dx_out + (*v_jac[inDim][secondDim])*(*ddb);
         }
      }

      if ((size_z > 1) && !zero_jac(inDim,thirdDim)) {
         /* Need d/dc */
         if (!b_dc) {
            calculate_d(type_z,T_c,ddc);
            b_dc = true;
         }
         if (constant_jac(inDim,thirdDim)) {
            *dx_out = *dx_out + c_jac[inDim][thirdDim]*(*ddc);
         } else {
            *dx_out = *dx_out + (*v_jac[inDim][thirdDim])*(*ddc);
         }
      }
   }
   void Grad::get_dx(DTArray * restrict dx_out,bool accum) {
      /* Take the x-derivative.  This may involve derivatives in any
         or all of a/b/c numerical directions */
      
      if (!accum) { /* Not accumulating in the destination */
         *dx_out = 0; // Start with a zero derivative
      }

      do_diff(dx_out,firstDim);
   }

   void Grad::get_dy(DTArray * restrict dy_out, bool accum) {
      if (!accum) *dy_out = 0;
      do_diff(dy_out,secondDim);
   }

   void Grad::get_dz(DTArray * restrict dz_out, bool accum) {
      if (!accum) *dz_out = 0;
      do_diff(dz_out,thirdDim);
   }

   Trans1D * Grad::get_trans(Dimension dim) const {
      switch(dim) {
         case firstDim: return T_a;
         case secondDim: return T_b;
         case thirdDim: return T_c;
         default: abort();
      }
   }

   S_EXP Grad::get_type(Dimension dim) const {
      switch (dim) {
         case firstDim: return type_x;
         case secondDim: return type_y;
         case thirdDim: return type_z;
         default: abort();
      }
   }
      
}


      
