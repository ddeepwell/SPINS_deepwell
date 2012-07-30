#ifndef GRAD_HPP
#define GRAD_HPP 1

/* grad.hpp -- header file for the gradient wrapper, which deals with the
   intricacies of Jacobian mapping to allow derivatives in physical
   dimensions when coordinate mapping couples dimensions on the
   "numerical box"

   As an implementation note, the Jacobian here is stored as the form
   d(a,b,c)/d(x,y,z), so that d/dx is a sum of da/dx*d/da + db/dx*d/db +
   dc/dx*d/dc, each individual term of which is known */

#include "TArray.hpp"
#include "Parformer.hpp"
#include <blitz/array.h>

namespace TArrayn {
   /* Include in the TArrayn namespace, since this is used much like
      deriv_cheb / deriv_fft / etc */

   class Grad {
      public:
         /* Constructor with sizes, types, and communicator (default world) */
         Grad(int szx, int szy, int szz, 
               Transformer::S_EXP tx, 
               Transformer::S_EXP ty, 
               Transformer::S_EXP tz,
               MPI_Comm c = MPI_COMM_WORLD);
         /* Destructor */
         ~Grad();

         /* Set the Jacobian, either as a constant or to an array for
            varying Jacobians */
         void set_jac(Dimension d1, Dimension d2, double constant,
               blitz::Array<double,3> * varying=0);
         
         /* Get the Jacobian */
         void get_jac(Dimension d1, Dimension d2, double & constant,
               DTArray * & varying) const;
         
         /* Shortcut to find out if all Jacobian terms are constant */
         bool constant_jac() const;
         bool constant_diagonal_jac() const; // and whether it's of diagonal form
         /* Or only a single entry */
         bool constant_jac(Dimension d1, Dimension d2) const;
         /* And whether that entry is uniformly zero */
         bool zero_jac(Dimension d1, Dimension d2) const;
         bool diagonal_jac() const;

         /* Get sizes of the referred array */
         void get_sizes(int & szx, int & szy, int & szz) const;
         MPI_Comm get_communicator() const;

         /* Setup a given array for the grad operation */
         void setup_array(DTArray * ar, 
               Transformer::S_EXP t1, Transformer::S_EXP t2,
               Transformer::S_EXP t3);

         /* Get d/dx(ar) */
         void get_dx(DTArray * restrict dx_out, bool accum=false);
         /* d/dy(ar) */
         void get_dy(DTArray * restrict dy_out, bool accum=false);
         /* d/dz(ar) */
         void get_dz(DTArray * restrict dz_out, bool accum=false);

         /* Get access to the 1D spectral transforms directly */
         Transformer::Trans1D * get_trans(Dimension dim) const;

         Transformer::S_EXP get_type(Dimension dim) const;

      private:
         /* Sizes */
         int size_x, size_y, size_z;
         /* Types */
         Transformer::S_EXP type_x, type_y, type_z;
         /* Communicator */
         MPI_Comm comm;

         /* 2D array for non-varying Jacobian terms*/
         double c_jac[3][3];
         /* 2D array (of array pointers) for varying Jacobian terms */
         DTArray * restrict v_jac[3][3];

         /* Lower bounds and extents for allocation of arrays */
         blitz::TinyVector<int,3> a_lbound, a_extent;
         /* Storage Order for same */
         blitz::GeneralArrayStorage<3> a_storage;

         /* The array we're currently working on (not owned) */
         DTArray * my_array;

         /* Arrays for storage of d/da, d/db, d/dc */
         blitz::Array<double,3> * restrict dda, * restrict ddb, * restrict ddc;
         /* Booleans to see if we've already done a required derivative */
         bool b_da, b_db, b_dc;
         
         /* TransWrappers for respective transforms */
         Transformer::Trans1D *T_a, *T_b, *T_c;

         /* Helper function to calculate the proper numerical derivative */
         void calculate_d(Transformer::S_EXP type,
               Transformer::Trans1D * tform, blitz::Array<double,3> * dest);

         void do_diff(DTArray * restrict dx_out, Dimension inDim);

   };

}

#endif
