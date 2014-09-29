/* Parformer.hpp -- intended to be a drop-in parallel replacement for 
   Transformer / Transwrapper.  Developed seperately to keep myself from
   doing anything stupid in the meantime. */

#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP 1
#include <blitz/array.h>
#include "TArray.hpp"
#include "Splits.hpp"

namespace Transformer {
   using TArrayn::firstDim; using TArrayn::secondDim; using TArrayn::thirdDim;
   using TArrayn::Dimension;
   using TArrayn::CTArray; using TArrayn::DTArray;
   using TArrayn::TArray;
   using blitz::Array;
   using blitz::TinyVector;
   using blitz::GeneralArrayStorage;

   static const char *S_EXP_NAME[] = { "COSINE", "SINE", "FOURIER", "CHEBY", "REAL", "COMPLEX", "NONE" };
   enum S_EXP { /* Spectral expansions */
      COSINE = 0, // DCT, for no-normal boundaries
      SINE = 1, // DST, for derivatives of no-normal boundaries
      FOURIER = 2, // Fourier, for periodic domains
      CHEBY = 3, // Chebyshev basis
      REAL = 4, // Nonspecific real transform (cos/sin/cheby)
      COMPLEX = 5, // Nonspecific complex transform (fft)
      NONE = 6// No transform
   };

   class TransWrapper {
      /* Ideally, this class would be able to take real/complex/none
         as template parameters and present a wonderful, simple, unified,
         elegant interface that also makes your lunch and folds your laundry.

         This won't happen.  The transforming-arrays (TArray) class is
         intentionally broken into double and complex<double> specializations,
         so the only way of making an uber-interface here is to also wrap
         those interfaces.  And I don't want to write all that code twice.

         Instead, this code will have to keep pointers for seperate temporary
         arrays around. */

      public:
         TransWrapper(int szx, int szy, int szz, // Input sizes,
               S_EXP Ttx, S_EXP Tty, S_EXP Ttz, // Transform types,
               MPI_Comm c = MPI_COMM_WORLD); // MPI Communicator (world)

         TransWrapper(const TransWrapper & copyfrom); // copy constructor
         ~TransWrapper(); // Destructor

         /* Compute and return wavenumbers along the given transform
            dimension, given the last performed transform */
         double max_wavenum(Dimension dim);
         Array<double,1> wavenums(Dimension dim,int lb=0, int ub=0) const;

         /* Normalization factor, so that transform^-1 * transform is
            the identity */
         double norm_factor() const;

         /* Get the respective temporary arrays, basaed on the last
            completed transform */
         bool use_complex() const;
         CTArray * get_complex_temp() const ;
         DTArray * get_real_temp() const ;

         /* Forward and backward transforms */
         /* We allow re-specification of the transform types here because
            (especially with real transforms) it's sometimes useful to
            change the transform type on the fly.  For example, finding
            du/dx might involving taking a DCT, multiplying by the wavenumber,
            and then taking the inverse DST. */
         void forward_transform(DTArray * in, S_EXP Ttx, S_EXP Tty, S_EXP Ttz);
         void back_transform(Array<double,3> * out, S_EXP Ttx,
                                 S_EXP Tty, S_EXP Ttz);

         void get_sizes(int & szx, int & szy, int & szz) const {
            szx = sx; szy = sy; szz = sz;
         }

         MPI_Comm get_communicator() const {
            return communicator;
         }

      protected:
         S_EXP Tx, Ty, Tz; // Transform types in X, Y, Z dimensions
         int sx, sy, sz; // Global sizes in x, y, z, dimensions
         MPI_Comm communicator;

         /* Transposer pointers, allocated as necessary */
         Transposer<double> * realTransposer; // for all-real transforms
         Transposer<complex<double> > * complexTransposer; // complex transforms

         /* Temporary arrays */
         int * refcount; // Reference count, for sharing of temporaries

         /* Further implementation note: look into making these temporaries
            cached/static scope */
         
         DTArray * yz_real_temp; // Temporary for a y/z real transform
         DTArray * x_real_temp; // Temporary for a x-real transform

         CTArray * yz_compl_temp; // Temporary for y/z complex transform
         CTArray * x_compl_temp; // x-complex temporary

         // We'll need to keep local array sizes around and compute them
         // for temporary allocation during object construction.

         TinyVector<int,3> yz_real_lbound, yz_real_extent,
            x_real_lbound, x_real_extent,
            x_compl_lbound, x_compl_extent,
            yz_compl_lbound, yz_compl_extent;

         GeneralArrayStorage<3> yz_real_storage, yz_compl_storage,
            x_real_storage, x_compl_storage;
   };
   /* Now, specialize the TransWrapper class for 1D transforms, as needed
      by derivatives.  It's tedious, error-prone, and ugly to require
      other code to use transform(&src,NONE,COSINE,NONE), when we can
      just specify a dimension at object cretion time and be done with it. */

   class Trans1D: protected TransWrapper {
      public: 
         Trans1D(int szx, int szy, int szz, Dimension dim, S_EXP type,
               MPI_Comm c = MPI_COMM_WORLD);
         Trans1D(const Trans1D & copyfrom);
         ~Trans1D() {};

         double max_wavenum();
         Array<double,1> wavenums(); // No need to specify dimension
         Dimension getdim(); // Accessor for dimension

         /* Some functions can be exported wholesale */
         using TransWrapper::use_complex;
         using TransWrapper::get_complex_temp;
         using TransWrapper::get_real_temp;
         using TransWrapper::norm_factor;
         using TransWrapper::get_sizes;

         /* And simplified transform specs */
         void forward_transform(DTArray * in, S_EXP type);
         void back_transform(Array<double,3> * out, S_EXP type);

      protected:
         Dimension trans_dim;
   };


} // End namespace


#endif // PARFORMER
