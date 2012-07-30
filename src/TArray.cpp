#include "TArray.hpp"
#include "Plan_holder.hpp"
#include <blitz/tinyvec-et.h>
#include <assert.h>
#include <fftw3.h> // FFTW
/* Implementation of TArray transform functions for the relevant cases.
   A truly general implementation is not possible because fftw:
      1) Only provides a limited set of implementations itself, since it is
      a C library, and
      2) Uses quite distinct function calls for different types of transforms
      (real-to-complex, complex-to-complex, cosine/sine), even ignoring type
      differences.  
   Most of the useful transforms only make sense for certain datatypes.  For
   example, a cosine transform requires both the source and destination arrays
   be real-valued.  Thus, we take advantage of template specialization in 
   C++ to do some additional compile-time error checking.
*/
int plan_counts = 0;
namespace TArrayn {

using blitz::TinyVector;

//const int FFTW_PLAN_TYPE = FFTW_EXHAUSTIVE;
const int FFTW_PLAN_TYPE = FFTW_MEASURE;
//const int FFTW_PLAN_TYPE = FFTW_ESTIMATE;

/* Real-to-real transforms; currently only DCT is implemented.  A DST may
   eventually be useful. */
template<> template<> void DTArray::transform<double>(
      Array<double,3> & dest, Dimension dimension, 
      Trans trans_type) {
   /* First and foremostly, do error checking on everything possible. */

   /* Assert that the bounds and storage of the source and destination arrays
      are the same */
   assert(all(this->lbound() == dest.lbound()));
   assert(all(this->ubound() == dest.ubound()));
   assert(all(this->stride() == dest.stride()));
//   assert(!any(blitz_isnan(*this)));
   
   /* It is perhaps unnecessary, but assert that the storage is continguous */
   assert(this->isStorageContiguous() && dest.isStorageContiguous());

   assert(trans_type == DCT0 || trans_type == DCT1 || trans_type == IDCT1
         || trans_type == DST1 || trans_type == IDST1);

   /* Construct a plan_spec governing this transform, to see if we perhaps
      have the hard work (plan formation) done already */
   Plan_spec this_transform(this->data(),dest.data(),trans_type,dimension);
//   cout << "plans cache size: " << plans.size();
   p_hash::iterator seen = plans.find(this_transform);
   if (seen != plans.end()) { // We've seen this plan before
      seen->second.execute(); // Execute the plan
//      fprintf(stderr,"Plan executing: %ld\n",(&(seen->second)));
//      assert(!any(blitz_isnan(dest)));
      return;                 // and we're done.
   }
//   cout << "+[" << &plans <<  "," << this->data() <<  "]\n";

   plan_counts++;

   fftw_r2r_kind plan_type;
   switch (trans_type) {
      case DCT0:
         plan_type = FFTW_REDFT00;
         break;
      case DCT1:
         plan_type = FFTW_REDFT10;
         break;
      case IDCT1:
         plan_type = FFTW_REDFT01;
         break;
      case DST1:
         plan_type = FFTW_RODFT10;
         break;
      case IDST1:
         plan_type = FFTW_RODFT01;
         break;
      default:
         abort();
   }
   /* This is a novel transform, one we haven't seen before.  Build the
      FFTW plans */
   if (this->stride(dimension) == max(this->stride())) {
      /* If this stride is the largest, it means we can do the transform with
         a single FFTW plan.  Each element is separated by stride(dimension),
         and each array is a "distance" of min(stride()) apart. */
      int stride = this->stride(dimension);
      int num_transforms = this->numElements() / this->extent(dimension);
      int distance = this->stride(this->ordering(0));
      int n = this->extent(dimension);

      /* When creating plans with anything other than FFTW_ESTIMATE, the
         plan creation is destructive of the contents of the array.  Since
         we don't want to do a transform of gibberish, we need to copy our
         array's contents someplace safe. */

      Baseclass temp = this->copy();


      fftw_plan major_plan = fftw_plan_many_r2r(
            1, &n, num_transforms,
            this->data(), 0, stride, distance,
            dest.data(), 0, stride, distance,
            &plan_type, FFTW_PLAN_TYPE);
      assert(major_plan); // Assert that the plan indeed was created

      (*this) = temp; // Copy data back over;

      // Store the plan in the cache and execute
      plans[this_transform] = Plan_holder(major_plan);
      plans[this_transform].execute(); 
//      fprintf(stderr,"Plan made and executing: %ld\n",(&plans[this_transform]));
      return;
   }
   else if (abs(this->stride(dimension)) ==
         abs(this->stride(this->ordering(0)))) {
      /* Minimum stride case -- we have a number of 1D arrays to transform, and
         the beginning of the next is located immediately after the end of
         the previous */
      int stride = this->stride(dimension);
      int num_transforms = this->numElements() / this->extent(dimension);
      int distance = this->extent(dimension);
      int n = distance;

      Baseclass temp = this->copy();


      fftw_plan major_plan = fftw_plan_many_r2r(
            1, &n, num_transforms,
            this->data(), 0, stride, distance,
            dest.data(), 0, stride, distance,
            &plan_type, FFTW_PLAN_TYPE);
      assert(major_plan);

      *this = temp;
      plans[this_transform] = Plan_holder(major_plan);
      plans[this_transform].execute();
//      fprintf(stderr,"Plan made and executing: %ld\n",(&plans[this_transform]));
      return;
   }
   else {
      /* General case: intermediate stride.  Unfortunately, FFTW3 doesn't
         provide a simple, one-plan solution for this, so we have to build
         a number of plans.  Each will transform a single (memory-contiguous)
         2D slice of the 3D array. */
      assert(dimension == this->ordering(1)); // Sanity check

      int stride = this->stride(dimension);
      int largedim = this->ordering(2); // Major dimension
      int smalldim = this->ordering(0); // Minor dimension

      int num_transforms = this->extent(smalldim);
      int distance = this->stride(smalldim);
      int n = this->extent(dimension);

      Baseclass temp = this->copy();
      vector<fftw_plan> planvec; // Vector to hold all of our generated plans

      TinyVector<int,3> base; // TinyVector to hold the index of our transform
      base(smalldim) = this->lbound(smalldim);
      base(dimension) = this->lbound(dimension);

      /* Now, loop over all of our 2D planes and build the FFTW plans.
         As a schematic example, we have a plan for each of (0,0,0), (0,0,1),
         etc -- corresponding to the planes (x,y,0), (x,y,1), etc. */
      for (int kk = this->lbound(largedim);
            kk <= this->ubound(largedim); kk++) {
         base(largedim) = kk; 
         fftw_plan major_plan = fftw_plan_many_r2r(
               1, &n, num_transforms,
               &((*this)(base)), 0, stride, distance,
               &(dest(base)), 0, stride, distance,
               &plan_type, FFTW_PLAN_TYPE);
         assert(major_plan);

         planvec.push_back(major_plan);
      }

      *this = temp;

      plans[this_transform] = Plan_holder(planvec);
      plans[this_transform].execute();
//      fprintf(stderr,"Plan (array) made and executing: %ld\n",(&plans[this_transform]));
      return;
   }
}
typedef complex<double> cdouble;

/* Real to complex transform, for the FFT of a real-valued source */
template<> template<> void DTArray::transform<cdouble>(
      Array<cdouble,3> & dest, Dimension dimension,
      Trans trans_type) {
   /* Error checking is slightly more difficult with R2C transforms, because
      the sizes do not match up perfectly.  A (one-dimension) array of N
      real values transforms to floor(N/2+1) complex values -- the
      ``other half'' are fixed because of symmetry. */

   /* lbounds, however, should be the same. */
   assert(all(this->lbound() == dest.lbound()));
   assert(all(this->ordering() == dest.ordering()));
   assert(this->isStorageContiguous());
   assert(dest.isStorageContiguous());
   assert(trans_type == FFTR);
//   assert(!any(blitz_isnan(*this)));

   for(Dimension d = firstDim; d <= int(thirdDim); d=Dimension(int(d)+1)) {
      if (d != dimension) {
         assert(this->ubound(d) == dest.ubound(d));
      } else {
         /* The transformed dimension is halved as part of the real-to-
            complex transform.  A size 1 array (only DC) has a size 1
            output, a size 2 array (DC and Nyquist) has a size 2 output,
            a size 3 array (DC and two non-Nyquist) also has a siz 2 output
            (but the second frequency is not necessarily real), etc. */
         assert(this->lbound(d) == 0);
         assert(dest.lbound(d) == 0);
         assert(dest.ubound(d) == (this->ubound(d)+1)/2);
      }
   }
   /* End of error checking */

   Plan_spec this_transform(this->data(),dest.data(),trans_type,dimension);
   p_hash::iterator seen = plans.find(this_transform);
   if (seen != plans.end()) { // We've seen the plan before
      seen->second.execute();
      return;
   }

   plan_counts++;

      if (this->stride(dimension) == max(this->stride())) {
         /* Major dimension */
         int sstride = this->stride(dimension);
         int dstride = dest.stride(dimension);
         int num_transforms = this->numElements() / this->extent(dimension);
         int sdistance = this->stride(this->ordering(0));
         int ddistance = dest.stride(dest.ordering(0));
         int n = this->extent(dimension);

         Baseclass temp = this->copy();


         fftw_plan major_plan = fftw_plan_many_dft_r2c(
               1, &n, num_transforms,
               this->data(), 0, sstride, sdistance,
               reinterpret_cast<fftw_complex*>(dest.data()), 0, 
               dstride, ddistance, FFTW_PLAN_TYPE);
         assert(major_plan); // Assert that the plan was created

         (*this) = temp;

         plans[this_transform] = Plan_holder(major_plan);
         plans[this_transform].execute();
         return;
      }
      else if (this->stride(dimension) == this->stride(this->ordering(0))) {
         /* Minimum stride case */
         int sstride = this->stride(dimension);
         int dstride = dest.stride(dimension);
         int num_transforms = this->numElements() / this->extent(dimension);
         int sdistance = this->extent(dimension);
         int ddistance = dest.extent(dimension);
         int n = sdistance;

         Baseclass temp = this->copy();


         fftw_plan minor_plan = fftw_plan_many_dft_r2c(
               1, &n, num_transforms,
               this->data(), 0, sstride, sdistance,
               reinterpret_cast<fftw_complex*>(dest.data()), 0,
               dstride, ddistance, FFTW_PLAN_TYPE);
         assert(minor_plan);

         (*this) = temp;

         plans[this_transform] = Plan_holder(minor_plan);
         plans[this_transform].execute();
         return;
      } else { // Intermediate case
         assert(dimension == this->ordering(1));
         int sstride = this->stride(dimension);
         int dstride = dest.stride(dimension);
         int num_transforms = this->extent(this->ordering(0));
         int sdistance = this->stride(this->ordering(0));
         int ddistance = dest.stride(dest.ordering(0));
         int n = this->extent(dimension);

         int smalldim = this->ordering(0);
         int largedim = this->ordering(2);

         Baseclass temp = this->copy();
         vector<fftw_plan> planvec;

         TinyVector<int,3> base;
         base(smalldim) = this->lbound(smalldim);
         base(dimension) = this->lbound(dimension);
         
         for(int kk = this->lbound(largedim);
               kk <= this->ubound(largedim); kk++) {
            base(largedim) = kk;
            fftw_plan major_plan = fftw_plan_many_dft_r2c(
                  1, &n, num_transforms,
                  &((*this)(base)), 0, sstride, sdistance,
                  reinterpret_cast<fftw_complex*> (&dest(base)),0,
                  dstride, ddistance, FFTW_PLAN_TYPE);
            assert(major_plan);
            planvec.push_back(major_plan);
         }

         (*this) = temp;

         plans[this_transform] = Plan_holder(planvec);
         plans[this_transform].execute();
         return;
      }
}

/* Complex to real transforms, adapted from real-to-complex above */
template<> template<> void CTArray::transform<double>(
      Array<double,3> & dest, Dimension dimension,
      Trans trans_type) {
   /* Error checking is slightly more difficult with C2R transforms, because
      the sizes do not match up perfectly.  A (one-dimension) array of N
      real values transforms to floor(N/2+1) complex values -- the
      ``other half'' are fixed because of symmetry. */

   /* lbounds, however, should be the same. */
   assert(all(this->lbound() == dest.lbound()));
   assert(all(this->ordering() == dest.ordering()));
   assert(this->isStorageContiguous());
   assert(dest.isStorageContiguous());
   assert(trans_type == IFFTR);
//   assert(!any(blitz_isnan(real(*this))));
//   assert(!any(blitz_isnan(imag(*this))));

   for(Dimension d = firstDim; d <= int(thirdDim); d=Dimension(int(d)+1)) {
      if (d != dimension) {
         assert(this->ubound(d) == dest.ubound(d));
      } else {
         /* The transformed dimension is halved as part of the real-to-
            complex transform.  A size 1 array (only DC) has a size 1
            output, a size 2 array (DC and Nyquist) has a size 2 output,
            a size 3 array (DC and two non-Nyquist) also has a siz 2 output
            (but the second frequency is not necessarily real), etc. */
         assert(this->lbound(d) == 0);
         assert(dest.lbound(d) == 0);
         assert(this->ubound(d) == (dest.ubound(d)+1)/2);
      }
   }
   /* End of error checking */

   Plan_spec this_transform(this->data(),dest.data(),trans_type,dimension);
   p_hash::iterator seen = plans.find(this_transform);
   if (seen != plans.end()) { // We've seen the plan before
      seen->second.execute();
      return;
   }

   plan_counts++;

      if (this->stride(dimension) == max(this->stride())) {
         /* Major dimension */
         int sstride = this->stride(dimension);
         int dstride = dest.stride(dimension);
         int num_transforms = this->numElements() / this->extent(dimension);
         int sdistance = this->stride(this->ordering(0));
         int ddistance = dest.stride(dest.ordering(0));
         int n = dest.extent(dimension);

         Baseclass temp = this->copy();

         fftw_plan major_plan = fftw_plan_many_dft_c2r(
               1, &n, num_transforms,
               reinterpret_cast<fftw_complex*>(this->data()), 0, 
               sstride, sdistance,
               (dest.data()), 0, dstride, ddistance, 
               FFTW_PLAN_TYPE | FFTW_PRESERVE_INPUT);
         assert(major_plan); // Assert that the plan was created

         (*this) = temp;

         plans[this_transform] = Plan_holder(major_plan);
         plans[this_transform].execute();
         return;
      }
      else if (this->stride(dimension) == this->stride(this->ordering(0))) {
         /* Minimum stride case */
         int sstride = this->stride(dimension);
         int dstride = dest.stride(dimension);
         int num_transforms = this->numElements() / this->extent(dimension);
         int sdistance = this->extent(dimension);
         int ddistance = dest.extent(dimension);
         int n = ddistance;

         Baseclass temp = this->copy();

         fftw_plan minor_plan = fftw_plan_many_dft_c2r(
               1, &n, num_transforms,
               reinterpret_cast<fftw_complex*>(this->data()), 0, 
               sstride, sdistance,
               (dest.data()), 0, dstride, ddistance, 
               FFTW_PLAN_TYPE);
         assert(minor_plan);

         (*this) = temp;

         plans[this_transform] = Plan_holder(minor_plan);
         plans[this_transform].execute();
         return;
      } else { // Intermediate case
         assert(dimension == this->ordering(1));
         int sstride = this->stride(dimension);
         int dstride = dest.stride(dimension);
         int num_transforms = this->extent(this->ordering(0));
         int sdistance = this->stride(this->ordering(0));
         int ddistance = dest.stride(dest.ordering(0));
         int n = dest.extent(dimension);

         int smalldim = this->ordering(0);
         int largedim = this->ordering(2);

         Baseclass temp = this->copy();
         vector<fftw_plan> planvec;

         TinyVector<int,3> base;
         base(smalldim) = this->lbound(smalldim);
         base(dimension) = this->lbound(dimension);
         
         for(int kk = this->lbound(largedim);
               kk <= this->ubound(largedim); kk++) {
            base(largedim) = kk;
            fftw_plan major_plan = fftw_plan_many_dft_c2r(
                  1, &n, num_transforms,
                  reinterpret_cast<fftw_complex*> (&(*this)(base)), 0, 
                  sstride, sdistance,
                  (&dest(base)),0, dstride, ddistance, 
                  FFTW_PLAN_TYPE);
            assert(major_plan);
            planvec.push_back(major_plan);
         }

         (*this) = temp;

         plans[this_transform] = Plan_holder(planvec);
         plans[this_transform].execute();
         return;
      }
}

/* Complex->Complex FFT, based off of the DCT code.  Preconditions on array
   size remain the same, since unlike real-to-complex transforms a dimension
   is not halved in size. */
template<> template<> void CTArray::transform<cdouble>(
      Array<cdouble,3> & dest, Dimension dimension, 
      Trans trans_type) {
   /* First and foremostly, do error checking on everything possible. */

   /* Assert that the bounds and storage of the source and destination arrays
      are the same */
   assert(all(this->lbound() == dest.lbound()));
   assert(all(this->ubound() == dest.ubound()));
   assert(all(this->stride() == dest.stride()));
//   assert(!any(blitz_isnan(real(*this))));
//   assert(!any(blitz_isnan(imag(*this))));
   
   /* It is perhaps unnecessary, but assert that the storage is continguous */
   assert(this->isStorageContiguous() && dest.isStorageContiguous());

   assert(trans_type == FFT || trans_type == IFFT);

   /* Construct a plan_spec governing this transform, to see if we perhaps
      have the hard work (plan formation) done already */
   Plan_spec this_transform(this->data(),dest.data(),trans_type,dimension);
   p_hash::iterator seen = plans.find(this_transform);
   if (seen != plans.end()) { // We've seen this plan before
      seen->second.execute(); // Execute the plan
      return;                 // and we're done.
   }

   plan_counts++;

   int trans_sign;
   /* Check for forward or backward FFT */
   if (trans_type == FFT)
      trans_sign = FFTW_FORWARD;
   if (trans_type == IFFT)
      trans_sign = FFTW_BACKWARD;
   /* This is a novel transform, one we haven't seen before.  Build the
      FFTW plans */
   if (this->stride(dimension) == max(this->stride())) {
      /* If this stride is the largest, it means we can do the transform with
         a single FFTW plan.  Each element is separated by stride(dimension),
         and each array is a "distance" of min(stride()) apart. */
      int stride = this->stride(dimension);
      int num_transforms = this->numElements() / this->extent(dimension);
      int distance = this->stride(this->ordering(0));
      int n = this->extent(dimension);

      /* When creating plans with anything other than FFTW_ESTIMATE, the
         plan creation is destructive of the contents of the array.  Since
         we don't want to do a transform of gibberish, we need to copy our
         array's contents someplace safe. */

      Baseclass temp = this->copy();

      fftw_plan major_plan = fftw_plan_many_dft(
            1, &n, num_transforms,
            reinterpret_cast<fftw_complex*>(this->data()), 0, 
            stride, distance,
            reinterpret_cast<fftw_complex*>(dest.data()), 0, 
            stride, distance,
            trans_sign, FFTW_PLAN_TYPE);
      assert(major_plan); // Assert that the plan indeed was created

      (*this) = temp; // Copy data back over;

      // Store the plan in the cache and execute
      plans[this_transform] = Plan_holder(major_plan);
      plans[this_transform].execute(); 
      return;
   }
   else if (abs(this->stride(dimension)) ==
         abs(this->stride(this->ordering(0)))) {
      /* Minimum stride case -- we have a number of 1D arrays to transform, and
         the beginning of the next is located immediately after the end of
         the previous */
      int stride = this->stride(dimension);
      int num_transforms = this->numElements() / this->extent(dimension);
      int distance = this->extent(dimension);
      int n = distance;

      Baseclass temp = this->copy();

      fftw_plan major_plan = fftw_plan_many_dft(
            1, &n, num_transforms,
            reinterpret_cast<fftw_complex*>(this->data()), 0, 
            stride, distance,
            reinterpret_cast<fftw_complex*>(dest.data()), 0, 
            stride, distance,
            trans_sign, FFTW_PLAN_TYPE);
      assert(major_plan);

      *this = temp;
      plans[this_transform] = Plan_holder(major_plan);
      plans[this_transform].execute();
      return;
   }
   else {
      /* General case: intermediate stride.  Unfortunately, FFTW3 doesn't
         provide a simple, one-plan solution for this, so we have to build
         a number of plans.  Each will transform a single (memory-contiguous)
         2D slice of the 3D array. */
      assert(dimension == this->ordering(1)); // Sanity check

      int stride = this->stride(dimension);
      int largedim = this->ordering(2); // Major dimension
      int smalldim = this->ordering(0); // Minor dimension

      int num_transforms = this->extent(smalldim);
      int distance = this->stride(smalldim);
      int n = this->extent(dimension);

      Baseclass temp = this->copy();
      vector<fftw_plan> planvec; // Vector to hold all of our generated plans

      TinyVector<int,3> base; // TinyVector to hold the index of our transform
      base(smalldim) = this->lbound(smalldim);
      base(dimension) = this->lbound(dimension);

      /* Now, loop over all of our 2D planes and build the FFTW plans.
         As a schematic example, we have a plan for each of (0,0,0), (0,0,1),
         etc -- corresponding to the planes (x,y,0), (x,y,1), etc. */
      for (int kk = this->lbound(largedim);
            kk <= this->ubound(largedim); kk++) {
         base(largedim) = kk; 
         fftw_plan major_plan = fftw_plan_many_dft(
               1, &n, num_transforms,
               reinterpret_cast<fftw_complex*>(&((*this)(base))), 0, 
               stride, distance,
               reinterpret_cast<fftw_complex*>(&(dest(base))), 0, 
               stride, distance,
               trans_sign, FFTW_PLAN_TYPE);
         assert(major_plan);

         planvec.push_back(major_plan);
      }

      *this = temp;

      plans[this_transform] = Plan_holder(planvec);
      plans[this_transform].execute();
      return;
   }
}

Trans invert_trans(Trans in) {
   /* Helper function to return proper inverse transform for a given type */
   switch (in) {
      case DCT0:
         return DCT0;
      case DCT1:
         return IDCT1;
      case IDCT1:
         return DCT1;
      case DST1:
         return IDST1;
      case IDST1:
         return DST1;
      case FFTR:
         return IFFTR;
      case IFFTR:
         return FFTR;
      case FFT:
         return IFFT;
      case IFFT:
         return FFT;
      default:
         abort();
   }
}

} // end namespace
