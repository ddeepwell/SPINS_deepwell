#include "Transformer.hpp"
#include "TArray.hpp"
#include <iostream>

using namespace std;

/* Implementation of TransWrapper, the abstraction class for Physical->Spectral
   transforms */

namespace Transformer {
//   using namespace TArray;
   using blitz::Array;

   /* Constructor */
   TransWrapper::TransWrapper(int szx, int szy, int szz, // sizes
         S_EXP dim1, S_EXP dim2, S_EXP dim3):  // expansions, by dimension
      real_temp(0), real_temp_rc(0), complex_temp(0), complex_temp_rc(0),
      allNone(true), r2c_dim(firstDim)  {
         sizes[firstDim] = szx;
         sizes[secondDim] = szy;
         sizes[thirdDim] = szz;
         /* If a size is one, force a NONE-type transform, since anything
            else doesn't make sense */
         if (szx == 1) dim1 = NONE;
         if (szy == 1) dim2 = NONE;
         if (szz == 1) dim3 = NONE;
         compute_ordering(dim1, dim2, dim3); // Compute transform order
         alloc_real(); // Allocate real temporary array, if necessary
         alloc_complex(); // Allocate complex temporary, if necessary
      }

   /* Copy constructor */
   TransWrapper::TransWrapper(const TransWrapper & copyfrom):
      /* Most importantly, copy over temporary pointers and reference counts */
      real_temp(copyfrom.real_temp), real_temp_rc(copyfrom.real_temp_rc),
      complex_temp(copyfrom.complex_temp), 
      complex_temp_rc(copyfrom.complex_temp_rc)
      {
         /* Copy private data over.  */
         sizes[firstDim] = copyfrom.sizes[firstDim];
         sizes[secondDim] = copyfrom.sizes[secondDim];
         sizes[thirdDim] = copyfrom.sizes[thirdDim];
         transform_types[firstDim] = copyfrom.transform_types[firstDim];
         transform_types[secondDim] = copyfrom.transform_types[secondDim];
         transform_types[thirdDim] = copyfrom.transform_types[thirdDim];

         /* This is calculated by ordering, but there's no need to recompute */
         use_real[firstDim] = copyfrom.use_real[firstDim];
         use_real[secondDim] = copyfrom.use_real[secondDim];
         use_real[thirdDim] = copyfrom.use_real[thirdDim];
         ordering[firstDim] = copyfrom.ordering[firstDim];
         ordering[secondDim] = copyfrom.ordering[secondDim];
         ordering[thirdDim] = copyfrom.ordering[thirdDim];
         r2c_dim = copyfrom.r2c_dim;
         
         alloc_real();
         alloc_complex();
      }

   /* Destructor -- clean up temporaries */
   TransWrapper::~TransWrapper() {
      if (real_temp) { // Real temporary allocated
         (*real_temp_rc)--; // decrement reference
         if (!(*real_temp_rc)) { // If the reference is 0
            delete real_temp; // delete the temporary
            real_temp = 0;
            delete real_temp_rc; // and the reference counter
            real_temp_rc = 0;
         }
      } // Otherwise, we didn't allocate a real temporary
      if (complex_temp) { // repeat for complex temporary
         (*complex_temp_rc)--;
         if (!(*complex_temp_rc)) {
            delete complex_temp;
            complex_temp = 0;
            delete complex_temp_rc;
            complex_temp_rc = 0;
         }
      }
   }
   
   namespace {
      /* Helper function: compute the general transform type (REAL/COMPLEX/NONE)
         given a input type which may be specific.  Replaces multiple switch
         statements */
      S_EXP general_transform(S_EXP in) {
         switch(in) {
            case COSINE: // Real-trig expansions
            case SINE:
            case CHEBY: // Chebyshev is a mapped cosine expansion
            case REAL:
               return REAL;
            case FOURIER:
            case COMPLEX:
               return COMPLEX;
            case NONE:
               return NONE;
            default:
               abort(); //  Shouldn't get here.
         }
      }
   } // End anonymous namespace
         
   /* Compute the order in which to take transforms, whether to use the real
      or complex array, and which dimension (if any) gets halved in a
      real-to-complex transform */
   void TransWrapper::compute_ordering(S_EXP dim1, S_EXP dim2, S_EXP dim3) {
      /* First, generalize the transform types.  If we construct with a DCT
         specified, then there's no technical reason we can't transform with
         a DST later.  (Indeed, that will be highly useful since on a free-slip
         box the velocities are a mix of cosine and sine expansions). */
      /* Array to tell if we've assigned an order to this dimension */
      bool assigned[3] = {false, false, false}; 
      allNone = true; // Sanity check -- all None-type transforms is an error
      bool usingImag = false; // Whether we've made an imaginary transform yet
      int current_dim = 0;
      
      transform_types[firstDim] = general_transform(dim1);
      transform_types[secondDim] = general_transform(dim2);
      transform_types[thirdDim] = general_transform(dim3);

      /* Now, loop through the dimensions, and transform real dims first */
      for (int d = firstDim; d <= thirdDim; d++) {
         if (transform_types[d] == REAL || transform_types[d] == NONE) {
            assigned[d] = true;
            use_real[d] = true;
            ordering[current_dim] = Dimension(d);
            current_dim = current_dim + 1;
         }
         if (transform_types[d] == REAL)
            allNone = false;
      }
      /* Pick up any complex dimensions last */
      for (int d = firstDim; d <= thirdDim; d++) {
         if (transform_types[d] == COMPLEX) {
            assigned[d] = true;
            use_real[d] = false;
            if (!usingImag) { // First imaginary transform seen -- real2complex
               usingImag = true;
               r2c_dim = Dimension(d);
            }
            ordering[current_dim] = Dimension(d);
            current_dim = current_dim + 1;
            allNone = false;
         }
      }
      // Error checking
      assert(assigned[0] && assigned[1] && assigned[2]);
      //assert(!allNone);
   }

   /* Allocate real temporary, if used */
   void TransWrapper::alloc_real() {
      /* If the real temporary is used somewhere, then make sure it's allocated */
      if (  allNone || (use_real[0] && transform_types[0] != NONE) || 
            (use_real[1] && transform_types[1] != NONE) || 
            (use_real[2] && transform_types[2] != NONE)) { 
         if (real_temp){ // Have one already from copy constructor
            assert(real_temp_rc); // Reference should be allocated also
            *real_temp_rc += 1; // Increment the reference
         } else { // Allocate the temporary ourselves
            real_temp = new DTArray(sizes[0], sizes[1], sizes[2]);
            real_temp_rc = new int; // Don't forget the reference count
            *real_temp_rc = 1;
         }
      } else { // We're not using the real temporary.  Make sure it's -not- there.
         assert(!real_temp && !real_temp_rc);
      }
   }

   /* Likewise, for the complex temporary */
   void TransWrapper::alloc_complex() {
      if (!use_real[0] || !use_real[1] || !use_real[2]) {
         if (complex_temp) {
            assert(complex_temp_rc);
            *complex_temp_rc += 1;
         } else {
            int c_size[3] ; // Array sizes -- one dimension is reduced
            c_size[0] = sizes[0]; c_size[1] = sizes[1]; c_size[2] = sizes[2];
            /* The reduced dimension has floor(N/2 + 1) complex entries */
            c_size[r2c_dim] = (1+c_size[r2c_dim]/2);
            complex_temp = new CTArray(c_size[0], c_size[1], c_size[2]);
            complex_temp_rc = new int;
            *complex_temp_rc = 1;
         }
      } else { // Complex temporary is unused
         assert(!complex_temp && !complex_temp_rc);
      }
   }

   /* Whether user code should read the spectral data from the complex array */
   bool TransWrapper::use_complex() {
      return (!allNone && !use_real[ordering[2]]);
   }
   /* Pointer to the complex array */
   CTArray * TransWrapper::get_complex_temp() {
      /* It's an error to get the temporary when you're not supposed to */
      assert(use_complex()); 
      return complex_temp;
   }
   DTArray * TransWrapper::get_real_temp() {
      assert(!use_complex());
      return real_temp;
   }

   /* Finally, the beef -- actual computation of the forward transforms */
   void TransWrapper::forward_transform(DTArray * in,
         S_EXP dim1, S_EXP dim2, S_EXP dim3) {
      /* Zeroth, repeat the initial checking of array sizes.  A dimension
         with a size of 1 (reducing dimension, e.g. 2D as NxNx1) gets
         a transform type of NONE */
      if (sizes[firstDim] == 1) dim1 = NONE;
      if (sizes[secondDim] == 1) dim2 = NONE;
      if (sizes[thirdDim] == 1) dim3 = NONE;
      /* First, make sure the input transform parameters match the configuration:
         that is,  make sure we're not doing a FFT on a "REAL"-type transform or
         a real-trig transform on a "COMPLEX"-type. */
      assert(general_transform(dim1) == general_transform(transform_types[0]) &&
            general_transform(dim2) == general_transform(transform_types[1]) &&
            general_transform(dim3) == general_transform(transform_types[2]));

      if (allNone) {
         /* In the special case of no transform, copy the input to the real
            temporary array and return. */
         *real_temp = *in;
         return;
      }
      
      /* Now, record the transform we're doing.  This isn't necessary for
         the transform itself, but it means less complication when getting
         the wavenumbers. */
      transform_types[0] = dim1;
      transform_types[1] = dim2;
      transform_types[2] = dim3;

      /* Flag on whether we've done any transforms yet */
      bool transformed = false;

      /* Do the transform, dimension-by-dimension */
      for (int dd = 0; dd < 3; dd++) {
         Trans cur_trans; // Type (from TArray) of current transform
         switch(transform_types[ordering[dd]]) {
            case COSINE:
               cur_trans = DCT1; break;
            case SINE:
               cur_trans = DST1; break;
            case CHEBY:
               cur_trans = DCT0; break;
            case FOURIER:
               if (ordering[dd] == r2c_dim) {
                  cur_trans = FFTR;
               } else {
                  cur_trans = FFT;
               } break;
            case NONE:
               continue; // If no transform, go to the next dimension
            default:
               abort(); // Must give a specific, rather than general type.
         }
        if (use_real[ordering[dd]]) { // Real transform, so DCT/DST-type
           DTArray * from;
           /* If we've done a transform already (must have been real), then
              we want to transform what's already in the temporary array.
              Else, transform the array we were passed in. */
           if (transformed) from = real_temp;
           else from = in;
           from->transform(*real_temp, ordering[dd], cur_trans);
           transformed = true;
        }
        if (!use_real[ordering[dd]]) { // Complex transform
           /* There are two different types of complex transforms:
              1) is the real-to-complex, meaning we want to start with DTArray,
              2) is the complex-to-complex, CTArray throughout */
           
           if (ordering[dd] == r2c_dim) { // Real-to-complex transform
              DTArray * from;
              if (transformed) from = real_temp;
              else from = in;
              from->transform(*complex_temp, ordering[dd], cur_trans);
              transformed = true;
           } else { // complex-to-complex transform
              /* Going forward, a complex-to-complex transform can never be
                 the first transform, since we're given a -real- array. */
              assert(transformed); 
              complex_temp->transform(*complex_temp, ordering[dd], cur_trans);
           }
         }
      }
   }

   /* Based on forward_transform above, do the inverse problem -- take the 
      wavenumbers from the appopriate temporary, invert the transform as if
      it had been computed by the specified transform types.

      Why not just re-use the types given in the forward transforms?  First-
      derivatives.  Taking the derivative of a sine series (trig-sense) gives
      a cosine series, and vice versa -- we need to account for this. */
   void TransWrapper::back_transform(Array<double,3> * out,
         S_EXP dim1, S_EXP dim2, S_EXP dim3) {
      /* Zeroth, repeat the initial checking of array sizes.  A dimension
         with a size of 1 (reducing dimension, e.g. 2D as NxNx1) gets
         a transform type of NONE */
      if (sizes[firstDim] == 1) dim1 = NONE;
      if (sizes[secondDim] == 1) dim2 = NONE;
      if (sizes[thirdDim] == 1) dim3 = NONE;
      // Sanity checking, copied from forward_transform
      assert(general_transform(dim1) == general_transform(transform_types[0]) &&
            general_transform(dim2) == general_transform(transform_types[1]) &&
            general_transform(dim3) == general_transform(transform_types[2]));

      if (allNone) {
         /* In the degenerate case of no acutal transform, the "transform"
            is just a copy of the real temporary */
         *out = *real_temp;
         return;
      }
      
      // Unlike forward_transform, we do NOT want to record the transform type.
      // We can also get away without a 'transformed' check.

      /* Figure out the last transform that acually does anything */
      int lastTrans = 0;
      while (transform_types[ordering[lastTrans]] == NONE) lastTrans++;

      /* Reassign transform_types to reflect given transformations */
      transform_types[0] = dim1;
      transform_types[1] = dim2;
      transform_types[2] = dim3;

      /* Loop over wavenumbers, in reverse order */
      for (int dd = 2; dd >= 0; dd--) {
         Dimension mydim = ordering[dd];
         Trans cur_trans; 
         switch (transform_types[mydim]) { // Find out the TArray::transform type
            case COSINE:
               cur_trans = IDCT1; break;
            case SINE:
               cur_trans = IDST1; break;
            case CHEBY:
               cur_trans = DCT0; break;
            case FOURIER:
               if (mydim == r2c_dim) {
                  cur_trans = IFFTR;
               } else {
                  cur_trans = IFFT;
               } break;
            case NONE:
               continue;
            default:
               abort(); // Shouldn't reach here -- need a specific type
               cur_trans = IFFT; break;
         }
         if (use_real[mydim]) { // Real transform
            Array<double,3> * to;
            /* On the last backward transform, the output goes in *out */
            if (dd == lastTrans) to = out;
            else to = real_temp;
            real_temp->transform(*to,mydim,cur_trans);
         } else { // Complex transform
            if (mydim == r2c_dim) { // output is real
               Array<double,3> * to;
               if (dd == lastTrans) to = out;
               else to = real_temp;
               complex_temp->transform(*to,mydim,cur_trans);
            } else {// output is still complex
               assert(dd != 0); // This should -not- be the last inverse transform
               complex_temp->transform(*complex_temp,mydim,cur_trans);
            }
         }
      }
   }

   /* Return a 1D array of wavenumbers for dimension dim of the last forward
      transform */
   Array<double,1> TransWrapper::wavenums(Dimension dim) {
      Array<double,1> out;
      blitz::firstIndex ii;
      switch (transform_types[dim]) {
         case NONE: // Provided strictly for convenience.  No physical meaning
            out.resize(sizes[dim]);
            out = 0;
            return out;
         case SINE:
            out.resize(sizes[dim]);
            out = ii+1; // no 0-frequency for sine expansions
            return out;
         case COSINE:
         case CHEBY:
            /* The expressions are identical, but they have slightly different
               meanings between a normal Cosine transform and a Chebyshev-specific
               DCT0.  In the DCT0 case, the "wavenumber" is really the Chebyshev
               polynomial number, T_0 -> T_n */
            out.resize(sizes[dim]);
            out = ii;
            return out;
         case FOURIER:
            /* Two subcases -- r2c and c2c transforms */
            if (dim == r2c_dim) {
               out.resize(sizes[dim]/2+1);
               out = ii; // Simple case -- negative frequencies are removed
            } else { // Now with 100% more (negative) frequencies!
               out.resize(sizes[dim]);
               out = where(ii <= sizes[dim]/2, ii,
                     (ii-sizes[dim]));
            } return out;
         default:
            abort(); // Must call wavenums -after- a specific transform
      }
   }

   /* Compute and return the normalization factor to apply to a transformed
      array */
   double TransWrapper::norm_factor() {
      double factor = 1;
      for (int dim = 0; dim < 3; dim++) {
         switch(transform_types[dim]) {
            case NONE:
               break; // No normalization if no transform;
            case SINE: case COSINE: case CHEBY: case REAL:
               factor = factor * 2 * sizes[dim];
               break;
            case FOURIER: case COMPLEX:
               factor = factor * sizes[dim];
               break;
            default:
               abort(); // Shouldn't reach this
         }
      }
      return factor;
   }

   /* Specialized implementation for Trans1D, because derivatives only involve one
      transform at a time */

   Trans1D::Trans1D(int szx, int szy, int szz, // Array sizes
         Dimension dim, S_EXP type // transformed dimension, type
         ):
      TransWrapper(szx, szy, szz, // sizes
            (dim == firstDim ? type : NONE) , // And expand the (dim/type) to
            (dim == secondDim ? type : NONE) ,// a type per dimension
            (dim == thirdDim ? type : NONE)) {
         trans_dim = dim;
      }

   Trans1D::Trans1D(const Trans1D& copyfrom): // Copy constructor
      TransWrapper(copyfrom), trans_dim(copyfrom.trans_dim) {
      }

   Dimension Trans1D::getdim() {
      return trans_dim;
   }

   Array<double, 1> Trans1D::wavenums() {
      return TransWrapper::wavenums(trans_dim);
   }

   void Trans1D::forward_transform(DTArray * in, S_EXP type) {
      TransWrapper::forward_transform(in,
            (trans_dim == firstDim ? type : NONE),
            (trans_dim == secondDim ? type : NONE),
            (trans_dim == thirdDim ? type : NONE));
   }

   void Trans1D::back_transform(Array<double,3> * out, S_EXP type) {
      TransWrapper::back_transform(out,
            (trans_dim == firstDim ? type : NONE),
            (trans_dim == secondDim ? type : NONE),
            (trans_dim == thirdDim ? type : NONE));
   }

   Trans1D::~Trans1D() { } // Empty destructor
   
            
} // End namespace

