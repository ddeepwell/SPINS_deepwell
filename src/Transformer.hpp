/* Transformer.hpp -- header for class that abstracts out the mechanics of
   physical/spectral transforms, including proper handling of CTArray / DTArray
   (complex/real) array differences.

   This class is necessary to properly support FFT/DCT/DST-based fields with the
   same codebase, since the mathematical mechanics remain the same but datatypes
   and parameters change.  Also, CVT (Chebyshev Transform) dimensions should 
   integrate fairly well.

   With parallelization, this will be one of the classes that directly deals with
   the split array semantics.  If the transform is entirely Trig (Winters-model-
   esque) then it will be the only high-level parallel-aware class. */
#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1
#include <blitz/array.h>
#include "TArray.hpp"

namespace Transformer {
   using namespace TArrayn;
   using blitz::Array;

   enum S_EXP { /* Spectral Expansion */
      COSINE, // Discrete Cosine Transform 
      SINE, // Discrete Sine Transform
      FOURIER, // Fast Fourier Transform
      CHEBY, // Chebyshev (different cosine) Transform
      REAL, // Generalized real transform -- placeholder for DCT/DST/CVT
      COMPLEX, // Generalized complex transform -- only FFT, included for symmetry
      NONE // No transform
   };

   class TransWrapper {
      /* Unlike other code in Spins, this class is not templated.  The problems
         of the CTArray/DTArray type differences can't be solved elegantly
         (and moreover maintainably) with template semantics.  So, we'll have
         to accept the minor wart of explicitly checking for Complex/Real output
         (spectral space) arrays. */

      /* This class will require a fairly extensive rewrite for parallelization.
         Most importantly, we won't (necessarily) have just a single real
         or complex temporary array, since at least one transpose will change
         the (distributed) shape.

         Addiitonally, local and global array sizes (domains) will not be
         the same, which will impact wavenumber computation. */

      public: 
         TransWrapper(int szx, int szy, int szz, // array sizes
               S_EXP dim1, S_EXP dim2, S_EXP dim3 // expansions
               );
         TransWrapper(const TransWrapper & copy_from); // copy constructor
         ~TransWrapper(); // Destructor

         /* Compute and return wavenumbers along the given dimension, based
            on the last transformed array */
         Array<double,1> wavenums(Dimension dim);

         /* Compute and return the normalization factor to apply to a
            transformed array */
         double norm_factor();

         /* Calling code needs to know where to get the transform after it's
            completed. */
         bool use_complex(); // Use the complex temporary?
         CTArray * get_complex_temp();
         DTArray * get_real_temp();

         /* The meat of the matter, actual transforms */
         // Forward transform
         void forward_transform(DTArray * in, S_EXP dim1, S_EXP dim2, S_EXP dim3);
         // Backward transform
         void back_transform(Array<double,3> * out, S_EXP dim1, S_EXP dim2, S_EXP dim3);
      protected:
         /* Recorded transform types.  It's okay to construct this class with a 
            SINE expansion and then call it with a COSINE -- that's just what will 
            happen if you want to use the same codepath for, say, pressure and a 
            tracer.  What's not okay is to construct this code for a REAL-type 
            transform and call it with a COMPLEX-type.  This Shouldn't Happen 
            (because FFT and Real-Trig expansions don't match on the same 
            dmension), but this will catch those sorts of errors. */
         S_EXP transform_types[3]; 
         
         bool use_real[3]; // Whether the output of the Nth transform is real
         /* Ordering -- what dimension goes first?  Specifically, ordering[0]
            is the first transform done, ordering[2] is the last */
         Dimension ordering[3]; 
         Dimension r2c_dim; // Which dimension gets the real-to-complex (if any)

         DTArray * real_temp; // Real temporary array
         int * real_temp_rc; // Reference count for real temporary array
         CTArray * complex_temp; // Complex temporary array
         int * complex_temp_rc; // Reference count; complex
         /* Implementation note -- why reference-count the arrays?  This class
            is pretty inexpensive, so it makes structural sense for anything 
            needing a spectral-transform to have one around.  Still, the
            temporary arrays deserve reuse, hence the copy constructor will
            reusue them.  But since dellocation of TransWrappers might happen
            in a mixed order, we have to be careful about when we delete the
            temporaries.  Multiple deletions can cause crashes or Bad Things. */

         // NOTE: ranges for local (parallel-split) sizes added later
         int sizes[3]; // global array sizes

         bool allNone; // Flag for degenerate no-transforms

         // Compute the proper transform ordering, since we can't do Real transform
         // operations on a Complex array.
         void compute_ordering(S_EXP dim1, S_EXP dim2, S_EXP dim3);
         void alloc_real(); // Allocate the real temporary, if necessary 
         void alloc_complex(); // Allocate the complex temporary, if necessary
   };

   /* Trans1D: Derivative class of TransWrapper for the special case of 1D
      transformations.

      These sorts of transformations come up all the time in derivative computations,
      whereas general (triple) spectral transforms show up in (separable) elliptic
      solves.  Requiring numerical code to use transform(&src,NONE,FOURIER,NONE) or
      transform(&src,FOURIER,NONE,NONE), [etc] is just plain stupid when a much
      simpler syntax of transform(&src,dimension,FOURIER) would suffice.  Indeed,
      because of temporary array considerations, we can fix the dimension at 
      object creation time and do transform(&src,TYPE).

      By extending TransWrapper, we get to re-use the machinery of temporary array
      allocation and (later) parallelizing transposes. */
   class Trans1D: protected TransWrapper {
      public:
         Trans1D(int szx, int szy, int szz, // Array sizes
               Dimension dim, S_EXP type // dimension, type
               );
         Trans1D(const Trans1D & copy_from); // copy constructor
         ~Trans1D();

         Array<double,1> wavenums(); // wavenumbers along transformed dimension
         Dimension getdim(); // retrieve transformed dimension number

         /* These functions can be exported wholesale from TransWrapper */
         using TransWrapper::use_complex;
         using TransWrapper::get_complex_temp;
         using TransWrapper::get_real_temp;
         using TransWrapper::norm_factor;

         /* Simplified forward/backward syntax */
         void forward_transform(DTArray * in, S_EXP type);
         void back_transform(Array<double,3> * out, S_EXP type);
      protected:
         Dimension trans_dim;
   };
}
#endif // TRANSFORMER_HPP
