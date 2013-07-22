#ifndef TARRAY_HPP // Prevent double inclusion
#define TARRAY_HPP 1
#include <blitz/array.h>

// Beginning with the C++-0x standard, the GNU hash_map
// extension is deprecated.  However, C++-0x is not yet
// fully supported by default with gcc, requiring an
// additional compile-time flag, so this hash_map cannot
// yet be simply replaed with std::unordered_map.  Instead,
// let's undefine __DEPRECATED
#ifdef __DEPRECATED
   #define _OLD_DEPRECATED
   #undef __DEPRECATED
#endif
#include <ext/hash_map>
#ifdef _OLD_DEPRECATED
   #define __DEPRECATED
#endif

#include <fftw3.h>
#include <complex>
#include "Plan_holder.hpp"


namespace TArrayn {

using __gnu_cxx::hash_map;
using __gnu_cxx::hash;
using std::complex;
using blitz::Array;
using blitz::TinyVector;
using blitz::GeneralArrayStorage;


enum Dimension {
   /* Create an enum for Dimension, to allow for additional type checking.
      Blitz (and by inheritance SPINS) uses dimensions of base 0, so the
      "first" dimension is actually dimension 0.  It is all to easy to use
      a numeric "1", so being careful with the type here should allow the
      compiler to catch some mistakes. */
   firstDim = blitz::firstDim, // 0
   secondDim = blitz::secondDim, // 1
   thirdDim = blitz::thirdDim // 2
};

/* Enum for transform types.  Allows for additional type-safety, in that
   an integer value isn't wrongly used in TArray::transform. */
enum Trans {DCT0 = 0, // Discrete Cosine Transform, for Chebyshev series
      DCT1, // Discrete Cosine Transform (1), for even-symmetry
      IDCT1, // Inverse of above
      DST1, // Derivative of DCT1, also used for odd symmetry
      IDST1, // Inverse
      FFTR, // Real to Complex Transform (FFT)
      IFFTR, // Complex to Real Transform (FFT)
      FFT, // Complex to Complex Transform (forward)
      IFFT}; // Complex to Complex Transform (backward)
template<class T> class TArray: public Array<T,3> {
   /* Main (templated) array type, extending Blitz Arrays to enable
      fftw transforms */
   private:
      typedef hash_map<Plan_spec, Plan_holder, plan_hash> p_hash;
      p_hash plans;
      TArray(const TArray & copyfrom); // private copy constructor for error
   public:
      typedef Array<T,3> Baseclass; // Base class of TArray

      /* Incoporate the operator= behavior of the Blitz Array, since this
         is not automatically accesible from an inherited class */
      TArray & operator = (const TArray  & rhs) {
         this->Baseclass::operator=(rhs);
         return *this;
      }
      using Baseclass::operator=; 

      // From Blitz::Array constructor
      TArray(const Baseclass & copyfrom) 
         :Baseclass(copyfrom), plans() {
         }

      TArray(int x, int y, int z) // Default constructor
         :Baseclass(x,y,z), plans() {
            *this=0;
         }

      /* Range constructor, no Storage Order */
      TArray(const TinyVector<int,3> & lbound,
             const TinyVector<int,3> & ubound)
         :Baseclass(lbound,ubound), plans() {
            *this=0;
         };

      /* Fully general constructor */
      TArray(const TinyVector<int,3> & lbound,
             const TinyVector<int,3> & ubound,
             GeneralArrayStorage<3> storage)
         :Baseclass(lbound,ubound,storage), plans() {
            *this=0;
         };

      /* Because of the interactions of Real-to-Complex transforms, the
         transform function itsemf must be a templated function.  In
         particular, Real-to-Complex transforms require different sizes on
         the input and output arrays. */
      template <class U> void transform(Array<U,3> & dest,
                                       Dimension dimension,
                                       Trans trans_type);
};

typedef TArray<double> DTArray; // Double Transform Array
typedef TArray<complex<double> > CTArray; // Complex Transform Array

Trans invert_trans(Trans in); /* Proper inverted type for input transform */

}
#endif
