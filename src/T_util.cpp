#include "T_util.hpp"
#include "TArray.hpp"
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>
#include "Par_util.hpp"
#include <complex>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdio.h>

#include <sys/mman.h> // Memmap
#include <sys/types.h> // open
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>


namespace TArrayn {

using blitz::Array;
using blitz::TinyVector;
using blitz::Range;
using blitz::cast;
using std::string;
using std::ofstream;
using std::swap;
using std::cerr; using std::endl;
using namespace Transformer;

/* Compute the derivative of an array, using the Fourier
   transform.  This function assumes that the array is periodic over a
   physical domain of length 2*pi.  This is generally not true, but
   correcting for this is a simple division (/L) in the calling code, and
   best done there. */
void deriv_fft(DTArray & source, Trans1D & tform, Array<double, 3> & dest) {
   /* Sanity checking -- the source and destination arrays should have
      the same ranges, and the dimension of interest should start at 0  */

   Dimension dim = tform.getdim();

   assert(all(source.lbound() == dest.lbound()));
   assert(all(source.ubound() == dest.ubound()));

   /* Perform first FFT */
//   fprintf(stderr,"&%g&\n",pvmax(source));
   tform.forward_transform(&source, FOURIER);

   Array<double,1> kvec = tform.wavenums();

   // Get the complex temporary array from the transformer
   CTArray * temp = tform.get_complex_temp();
   //fprintf(stderr,"&%g&\n",psmax(max(abs(*temp))));
   assert(temp->lbound(dim) == 0);
   
   if (dest.extent(dim) % 2 == 0 && max(kvec) == tform.max_wavenum()) {
      /* If we have an even number of points, the Nyquist frequency is
         representable on the discrete grid.  Unfortunately, the -derivative-
         of the Nyquist frequency is purely imaginary, which we cannot
         store in our real-valued destination.  Thus, zero out this highest
         frequency. */
      kvec(temp->ubound(dim)) = 0;
   }

   /* After the Fourier transform, we can take the derivative by multiplying
      each wavenumber by I*k.  Since we're only working in one dimension
      here, we can do this multiplication a plane at a time. */
   /* The Blitz RectDomain class allows for simple specification of a slice */ 

   //fprintf(stderr,"&%g&\n",psmax(max(abs(*temp))));
   blitz::RectDomain<3> slice = temp->domain(); /* indexing */

   double norm = tform.norm_factor(); // Normalization constant

   for(int kk = temp->lbound(dim); kk <= temp->ubound(dim); kk++) {
      slice.lbound(dim) = kk; 
      slice.ubound(dim) = kk;
      (*temp)(slice) *= complex<double>(0,1*kvec(kk)/norm);
   }
   //fprintf(stderr,"&%g&\n",psmax(max(abs(*temp))));

   /* And take the inverse Fourier transform to the destination array */
   tform.back_transform(&dest, FOURIER);
   //fprintf(stderr,"&%g&\n",pvmax(dest));
}

/* Derivatives for the cosine and sine transforms are nearly the same, so
   to avoid wasted code they can both be rolled up into a single, "real-
   transform" derivative function. */
namespace { /* Nested anonymous namespace for file-local linkage */
inline void deriv_real(DTArray & source, Trans1D & tform, 
      blitz::Array<double, 3> & dest, S_EXP t_type) {

   assert(all(source.lbound() == dest.lbound()));
   assert(all(source.ubound() == dest.ubound()));

   /* Get transformed dimension */
   Dimension dim = tform.getdim();
   

   S_EXP b_type; // Backwards transform type
   switch (t_type) {
      case SINE:
         b_type = COSINE; break;
      case COSINE:
         b_type = SINE; break;
      default: // Shouldn't reach this
//         assert(0 && "Invalid real derivative transform specified");
         abort();
   }

   /* transform */
   tform.forward_transform(&source, t_type);

   blitz::firstIndex ii;
   Array<double, 1> kvec = tform.wavenums();

   DTArray & temp = *(tform.get_real_temp());
   assert(temp.lbound(dim) == 0);

//   std::cout << temp;

   blitz::RectDomain<3> slicel=temp.domain(), slicer=temp.domain();

   double norm = tform.norm_factor(); // Normalization constant
   if (t_type == COSINE) {
      /* The derivative of the cosine is a sine, and the constant frequency
         (first cosine term) is completely absent in the sine transform.
         Therefore, in the differentiation process we must shift our
         wavenumbers DOWN one. */
      for (int kk = 0; kk < temp.ubound(dim); kk++) {
         slicel.ubound(dim) = kk; slicel.lbound(dim) = kk;
         slicer.lbound(dim) = kk+1; slicer.ubound(dim) = kk+1;
         temp(slicel) = -temp(slicer) * kvec(kk+1) / norm;
      }
      /* And zero out the highest frequency, which does not have a
         corresponding cosine term */
      temp(slicer) = 0;
   } else { 
      /* Derivative of the sine is a cosine, and the problem is reversed.
         The highest sine term has no equivalent cosine, so that frequency
         is ignored.  In the differentiation process, the frequencies are
         shifted UP one (the constant term becomes zero). */
      for (int kk = temp.ubound(dim); kk > 0; kk--) {
         slicel.lbound(dim) = kk; slicel.ubound(dim) = kk;
         slicer.lbound(dim) = kk-1; slicer.ubound(dim) = kk-1;
         temp(slicel) = temp(slicer) * kvec(kk-1) / norm;
      }
      /* Zero out the constant frequency term */
      temp(slicer) = 0;
   }

   /* And transform into the destination array */
//   std::cout << temp;
   tform.back_transform(&dest, b_type);

} 
} /* End anonymous local namespace */

/* Now, the cosine and sine derivatives are just thin wrappers around the
   above deriv_real. */

void deriv_dct(DTArray & source, Trans1D & tform, Array<double, 3> & dest) {
   deriv_real(source,tform,dest,COSINE);
}

void deriv_dst(DTArray & source, Trans1D & tform, Array<double, 3> & dest) {
   deriv_real(source,tform,dest,SINE);
}

void deriv_cheb(DTArray & source, Trans1D & tform, Array<double,3> & dest) {
   /* Chebyshev differentiation */
   /* Unlike other differentiations, chebyshev differentiation is defined
      by a recurrance relation:

      D = d/dx(F)
      D_(N-1) = 0;
      D_k = D_(k+2) + 2*(k+1)*F_(k+1)

      modulo possible normalization at the zero frequency.  This implies
      that we're going to have to store a couple temporary array slices
      around -- for F_(k) and F_(k+1) */
   Dimension dim = tform.getdim();
   assert(all(source.lbound() == dest.lbound()));
   assert(all(source.ubound() == dest.ubound()));
   if (source.extent(dim) == 1) {
      dest = 0; return;
   }

   /* Transform to Chebyshev domain */
   tform.forward_transform(&source, CHEBY);
   /* Get wavenumbers */
//   Array<double,1> kvec(tform.wavenums());
   /* Note -- optimizing for 1D Chebyshevs, and getting the wavenumbers
      here every derivative involves mallocing and bad things.  We know
      what the wavenumbers actually will be, so just substitute into
      the expression */


   /* And normalization factor */
   double norm = tform.norm_factor();

   /* Get the real temporary array */
   DTArray & temparray = *(tform.get_real_temp());
//   cout << temparray;
#if 1
   /* Dealing with this temporary array as a giant 3D grid is
      elegant, but has crappy Blitz++ performance; accessing
      the array through a Domain/slice notation is inefficient because
      it constructs a temporary Array.  Instead, we'll transpose the
      temporary array to push the derivative-dimension to thirdDim,
      and then loop over each element. */
   int my_dims[3];
   switch (dim) {
      case firstDim:
         my_dims[2] = firstDim;
         my_dims[0] = secondDim;
         my_dims[1] = thirdDim;
         break;
      case secondDim:
         my_dims[2] = secondDim;
         my_dims[0] = firstDim;
         my_dims[1] = thirdDim;
         break;
      case thirdDim:
         my_dims[2] = thirdDim;
         my_dims[0] = firstDim;
         my_dims[1] = secondDim;
         break;
      default:
         abort();
   }
   Array<double,3> tview =
      temparray.transpose(my_dims[0],my_dims[1],my_dims[2]);
   const int lbound = tview.lbound(thirdDim),
       ubound = tview.ubound(thirdDim);
   for (int i = tview.lbound(firstDim); i <= tview.ubound(firstDim); i++) {
//      cerr << __FILE__ << ":" << __LINE__ << "\n";
      for (int j = tview.lbound(secondDim); j<= tview.ubound(secondDim); j++) {
//      cerr << __FILE__ << ":" << __LINE__ << "\n";
         /* First, filter the transformed coefficients in this dimension */
         /* The cosine-based transformation is not exact, and rounding error
            grows quite significantly with N.  Consider that if a boundary
            value were off by epsilon, then its first derivative is off
            by O(N^2)*epsilon simply because of grid spacing.  The second
            derivative (used for Laplacian) is therefore off by O(N^4).

            Hence, filter out any small coefficients. */
#if 0
         double local_max = max(abs(tview(i,j,blitz::Range::all())));
         tview(i,j,blitz::Range::all()) = 
            where(abs(tview(i,j,blitz::Range::all())) > 1e-15*local_max,
                  tview(i,j,blitz::Range::all()), 0);
#endif

         /* Take the derivative along this dimension */
         /* The recurrence relation is:
            D = C'
            D(end) = 0;
            D(end-1) = (end-1)*C(end)
            D(z) = D(z+2) + 2*K(z+1)*C(z+1)
         */

         double cplus1 = tview(i,j,ubound);
         double cplus0 = tview(i,j,ubound-1);
         tview(i,j,ubound) = 0;
         tview(i,j,ubound-1) = (ubound-1)/norm*cplus1;
         cplus1 = cplus0;
         for (int k = ubound-2; k>= lbound; k--) {
            cplus0 = tview(i,j,k);
            tview(i,j,k) = tview(i,j,k+2) + (2*(k+1))/norm*cplus1;
            cplus1 = cplus0;
         }
//      cerr << __FILE__ << ":" << __LINE__ << "\n";
         
      }
   }
   /* No further processing is necessary, since tview is merely a transformed
      -view- of the temporary array inside the Trans1D. */
         
         
//   cout << temparray; // DEBUG: REMOVE ME
#else
   /* Allocate two temporary slices */
   /* First, find the right slice */
   blitz::RectDomain<3> dslice = temparray.domain();
   /* And compress it down to two dimensions */
   dslice.lbound(dim) = 0; dslice.ubound(dim) = 0;
   
   /* Compute extents vector to initialize the two temporary slices,
      since there is no constructor for a RectDomain */
   blitz::TinyVector<double,3> extents = dslice.ubound() - dslice.lbound() + 1;

   blitz::Array<double,3> tslice1(dslice.lbound(), extents),
                          tslice2(dslice.lbound(), extents);

   /* "Step zero" is to filter the transformed coefficients.
      The cosine-based transform is not exact; terms may have an error
      of about machine-epsilon times the largest coefficient.  While
      this is perfectly okay normally, taking derivatives multiplies
      the high terms.

      In physical space, the transforms of f(x) and f(x)+random*epsilon
      are identical, for some particular essentailly-random distribution
      given by rounding error in the transform.  So, if we evaluate
      the derivative exactly (using Chebyshev-defined cardinal functions),
      the numerical derivative will differ from the true derivative by
      the derivative of this random noise.  But (switching to finite-
      differnce thinking for simplicity), the grid spacing near the
      boundaries is O(1/N^2), implying epsilon times the discrete delta
      function is magnified by O(N^2) in derivative!

      This is -catrastrophic- for large numbers of points, especially for
      second derivatives that are the bread-and-butter of a Poisson or
      Helmholtz solve.  (Second derivative error scales as O(N^4) --
      already a completely-well-resolved function has 0.1% error by 
      1000 points! 

      To remedy, threshold the transformed coefficients; any coefficient
      in the derivatived-direction that is below 10^-14 times the maximum
      (in absolute value) is set to precisely zero. */

   /* For our given direction, take the maximum absolute value and store
      in tslice1.  Then, threshold temparray based on that maximum. */
   blitz::firstIndex ii;
   blitz::secondIndex jj;
   blitz::thirdIndex kk;
   switch(dim) {
      case firstDim: 
         {
            /* Create a 2D view of the trivially-3d temporary */
            blitz::Array<double,2> 
               tslice2d(tslice1(0,blitz::Range::all(),blitz::Range::all()));
            /* Assign to it with a partial reduction.  We can't assign
               directly to the 3D array because Blitz only supports partial
               reductions on the last dimension */
            tslice2d = max(abs(temparray(kk,ii,jj)),kk);
            temparray = where(abs(temparray(ii,jj,kk)) > 1e-14*tslice2d(jj,kk),
                  temparray(ii,jj,kk),0);
         }
         break;

      case secondDim: 
         {
            blitz::Array<double,2> 
               tslice2d(tslice1(blitz::Range::all(),0,blitz::Range::all()));
            tslice2d = max(abs(temparray(ii,kk,jj)),kk);
            temparray = where(abs(temparray(ii,jj,kk)) > 1e-14*tslice2d(ii,kk),
                  temparray(ii,jj,kk),0);
         }
         break;
      case thirdDim: 
         {
            blitz::Array<double,2> 
               tslice2d(tslice1(blitz::Range::all(),blitz::Range::all(),0));
            tslice2d = max(abs(temparray(ii,jj,kk)),kk);
            temparray = where(abs(temparray(ii,jj,kk)) > 1e-14*tslice2d(ii,jj),
                  temparray(ii,jj,kk),0);
         }
         break;
   }
   tslice1 = 0;

   /* And pointers so that the slices can be accessed through indirection */
   blitz::Array<double,3> * ts_kp = &tslice1, // (k+1)
                          * ts_k  = &tslice2;  // (k)

   blitz::RectDomain<3> kslice = dslice; // k slice
   blitz::RectDomain<3> k2slice = dslice; // k+2 slice
   kslice.lbound(dim) = kslice.ubound(dim) = temparray.ubound(dim);

   /* First, copy the maximum wavenumber to the k+1 temporary */
      /* NOTE: The /2 is here as a normalization factor, since the last
         Chebyshev polynomial is a special and unique flower. */
   (*ts_kp)(dslice) = temparray(kslice);///2;
   /* And zero that part of the temparray, since the highest polynomial in
      the derivative is always zero */
   temparray(kslice) = 0;
   /* Also write out the first explicitly, since the k+2 element is referenced,
      and that won't exist for k-1 */
   kslice.lbound(dim) = kslice.ubound(dim) = temparray.ubound(dim)-1;
   (*ts_k)(dslice) = temparray(kslice);
   /* Normally there is a two here.  Skip that for a normalization factor,
      since the Nyquist frequency is a rare and special flower */
   temparray(kslice) = (kvec(temparray.ubound(dim)))*(*ts_kp)/norm;

   swap(ts_kp,ts_k);

   for (int k = temparray.ubound(dim)-2; k >= temparray.lbound(dim); k--) {
      /* Step from high to low frequencies */
      kslice.lbound(dim) = kslice.ubound(dim) = k;
      k2slice.lbound(dim) = k2slice.ubound(dim) = k+2;
      (*ts_k)(dslice) = temparray(kslice);

      temparray(kslice) = temparray(k2slice) + 2*(1+kvec(k))*(*ts_kp)/norm;
      swap(ts_kp,ts_k);
   }
//   cout << temparray; // DEBUG: REMOVE ME
   /* Done! (?) */
#endif
   tform.back_transform(&dest,CHEBY);
}
      
   
   

/* In-place spectral filtering, using an exponential filter (of given order)
   after a cutoff frequency (faction of Nyquist) */
void filter3(DTArray & source, TransWrapper & tform, 
      S_EXP dim1_type, S_EXP dim2_type, S_EXP dim3_type,
      double cutoff, double order, double strength) {

   if (strength == 0) return;
   bool implicit_les = false;
   if (strength < 0) implicit_les = true;
   blitz::firstIndex ii;
   blitz::secondIndex jj;
   blitz::thirdIndex kk;
   /* Transform */
   tform.forward_transform(&source,dim1_type, dim2_type, dim3_type);

   /* Get wavenumbers and find their maximums */
   Array<double,1> kvec = tform.wavenums(firstDim),
                   lvec = tform.wavenums(secondDim),
                   mvec = tform.wavenums(thirdDim);
   double kmax = tform.max_wavenum(firstDim), 
          lmax = tform.max_wavenum(secondDim),
          mmax = tform.max_wavenum(thirdDim);
	
   if (kmax == 0) kmax = 1;
   if (lmax == 0) lmax = 1;
   if (mmax == 0) mmax = 1;
////   fprintf(stderr,"Filter3: kmax %g, lmax %g, mmax %g\n",kmax,lmax,mmax);

   double kcut = cutoff*kmax, lcut = cutoff*lmax, mcut = cutoff*mmax;
//   fprintf(stderr,"Filter3: kcut %g, lcut %g, mcut %g\n",kcut,lcut,mcut);

   double norm = tform.norm_factor();

   double strength_x = strength, strength_y = strength, strength_z = strength;

   if (implicit_les) {
//	   fprintf(stderr,"Filter3: Implicit LES on\n");
      /* Find out:
            1) How strong our fields are at low frequencies (10%), less
            the zero frequency
            2) How strong the high-order noise is (> cutoff)
            3) What filter ratio (anisotropic) we need to make them alright.
      */
      double low_level=0; 
      double high_k=0, high_l=0, high_m=0, 
             high_kl=0, high_lm=0, high_km=0, 
             high_klm=0;
      MPI_Comm c = tform.get_communicator();
      if (tform.use_complex()) {
         CTArray & temp = *(tform.get_complex_temp());
         for (int i = temp.lbound(firstDim); i <= temp.ubound(firstDim); i++)
            for (int j = temp.lbound(secondDim); j <= temp.ubound(secondDim); j++)
               for (int k = temp.lbound(thirdDim); k <= temp.ubound(thirdDim); k++) {
                  if (i == 0 && j == 0 && k == 0) continue;
                  if (abs(kvec(i)) < 0.1*kmax &&
                      abs(lvec(j)) < 0.1*lmax &&
                      abs(mvec(k)) < 0.1*mmax)
                        low_level = fmax(low_level,abs(temp(i,j,k)));
                  if (abs(kvec(i)) > kcut) {
                     if (abs(lvec(j)) > lcut) {
                        if (abs(mvec(k)) > mcut)
                           high_klm = fmax(high_klm,abs(temp(i,j,k)));
                        else
                           high_kl = fmax(high_kl,abs(temp(i,j,k)));
                     } else if (abs(mvec(k)) > mcut) {
                        high_km = fmax(high_km,abs(temp(i,j,k)));
                     } else
                        high_k = fmax(high_k,abs(temp(i,j,k)));
                  } else if (abs(lvec(j)) > lcut) {
                     if (abs(mvec(k)) > mcut) {
                        high_lm = fmax(high_lm,abs(temp(i,j,k)));
                     } else
                        high_l = fmax(high_l,abs(temp(i,j,k)));
                  } else if (abs(mvec(k)) > mcut) {
                     high_m = fmax(high_m,abs(temp(i,j,k)));
                  }
               }
      } else {
         DTArray & temp = *(tform.get_real_temp());
         for (int i = temp.lbound(firstDim); i <= temp.ubound(firstDim); i++)
            for (int j = temp.lbound(secondDim); j <= temp.ubound(secondDim); j++)
               for (int k = temp.lbound(thirdDim); k <= temp.ubound(thirdDim); k++) {
                  if (i == 0 && j == 0 && k == 0) continue;
                  if (abs(kvec(i)) < 0.1*kmax &&
                      abs(lvec(j)) < 0.1*lmax &&
                      abs(mvec(k)) < 0.1*mmax)
                        low_level = fmax(low_level,abs(temp(i,j,k)));
                  if (abs(kvec(i)) > kcut) {
                     if (abs(lvec(j)) > lcut) {
                        if (abs(mvec(k)) > mcut)
                           high_klm = fmax(high_klm,abs(temp(i,j,k)));
                        else
                           high_kl = fmax(high_kl,abs(temp(i,j,k)));
                     } else if (abs(mvec(k)) > mcut) {
                        high_km = fmax(high_km,abs(temp(i,j,k)));
                     } else
                        high_k = fmax(high_k,abs(temp(i,j,k)));
                  } else if (abs(lvec(j)) > lcut) {
                     if (abs(mvec(k)) > mcut) {
                        high_lm = fmax(high_lm,abs(temp(i,j,k)));
                     } else
                        high_l = fmax(high_l,abs(temp(i,j,k)));
                  } else if (abs(mvec(k)) > mcut) {
                     high_m = fmax(high_m,abs(temp(i,j,k)));
                  }
               }
      }

         /* Find the maximum of all these over all processors */
         double values[8] = {low_level, high_k, high_l, high_m, high_kl, high_lm, high_km,high_klm};
         MPI_Allreduce(MPI_IN_PLACE,values,8,MPI_DOUBLE,MPI_MAX,c);
         /* And reassign to the descriptive variables */
         low_level = values[0];
         high_k = values[1];
         high_l = values[2];
         high_m = values[3];
         high_kl = values[4];
         high_lm = values[5];
         high_km = values[6];
         high_klm = values[7];
         if (0&&master()) {
            fprintf(stderr,"Implicit filtering values:\n low_level: %g, \nhigh_k: %g, high_l: %g, high_m: %g, \nhigh_kl: %g, high_lm: %g, high_km: %g, \nhigh_klm: %g\n",low_level,high_k,high_l,high_m,high_kl,high_lm,high_km,high_klm);
         }

         // Now, compute what sort of strengths are required to tame possible
         // oscillations

         /* If there is no low-frequency component, filtering is meaningless */
         if (low_level == 0) return;

         /* Compute strengths required to reduce high-frequency noise
            to 1e-6 times low-frequency levels */
         
         /* If this condition is already satisfied, strength_x (eg) will
            already be log(1e6) ~= -13.82.  If the condition is more than
            satisfied, strength_x < -13.82.

            For reference, log(10) = 2.30 */
         strength_x = -log((low_level)/(high_k+1e-100));
         strength_y = -log((low_level)/(high_l+1e-100));
         strength_z = -log((low_level)/(high_m+1e-100));

         /* Now, that takes care of only the "middles".  The corners
            of the spectral domain are unhandles.  In some testing, these
            "corner cases" (ha ha ha I made a funny) ended up being important */
         
         double strength_xy = -log((low_level)/(high_kl+1e-100));
         double strength_xz = -log((low_level)/(high_km+1e-100));
         double strength_yz = -log((low_level)/(high_km+1e-100));
         double strength_xyz = -log((low_level)/(high_klm+1e-100));

         /* If the corners are worse (larger) than the middles, smoothly
            increase the middle-strengths to match.  As a catch-all, also
            ensure that the edges are not much smaller than the corner
            after correction -- otherwise the "0 on edges, 1 in corner"
            case doesn't get filtered. */
         double diff = strength_xy - fmax(strength_x,strength_y);
         if (diff > 0) {
            strength_x += diff/2;
            strength_y += diff/2;
            strength_x = fmax(strength_x,strength_xy-2.3);
            strength_y = fmax(strength_y,strength_xy-2.3);
         }
         diff = strength_xz - fmax(strength_x,strength_z);
         if (diff > 0) {
            strength_x += diff/2;
            strength_z += diff/2;
            strength_x = fmax(strength_x,strength_xz-2.3);
            strength_z = fmax(strength_z,strength_xz-2.3);
         }
         diff = strength_yz - fmax(strength_z,strength_y);
         if (diff > 0) {
            strength_z += diff/2;
            strength_y += diff/2;
            strength_y = fmax(strength_y,strength_yz-2.3);
            strength_z = fmax(strength_z,strength_yz-2.3);
         }
         diff = strength_xyz - fmax(strength_x,fmax(strength_y,strength_z));
         if (diff > 0) {
            strength_x += diff/3;
            strength_y += diff/3;
            strength_z += diff/3;
            strength_x = fmax(strength_x,strength_xyz-2.3);
            strength_y = fmax(strength_y,strength_xyz-2.3);
            strength_z = fmax(strength_z,strength_xyz-2.3);
         }

         /* Now, "smoothly turn-on" the filtering.  This affects timestepped
            variables, so a hard on/off filter might cause temporal
            oscillations */

         /* For high strength, we want to filter as approximately
            CONSTANT*(strength+13.82), dropping smoothly to 0 as
            the strength goes below 13.82.  Integrating 1+tanh
            is therefore probably our best bet. */

         /* Why the if statements?  cosh(x) is exponentially big, and ln(x)
            brings it back to a linear scale.  Since we don't want to mess up
            the floating point math, include end-caps */
         if (strength_x < -20)
            strength_x = 0;
         else if (strength_x > 0)
            strength_x = -strength*(strength_x+13.82);
         else
            strength_x = -strength*(2.3*log(2*cosh((strength_x+13.82)/2.3))+strength_x+13.82);
         if (strength_y < -20)
            strength_y = 0;
         else if (strength_y > 0)
            strength_y = -strength*(strength_y+13.82);
         else
            strength_y = -strength*(2.3*log(2*cosh((strength_y+13.82)/2.3))+strength_y+13.82);
         if (strength_z < -20)
            strength_z = 0;
         else if (strength_z > 0)
            strength_z = -strength*(strength_z+13.82);
         else
            strength_z = -strength*(2.3*log(2*cosh((strength_z+13.82)/2.3))+strength_z+13.82);

         /* Center the filter at 1e-6, and send the strength to 0 at 1e-7.
            By 1e-5, the filter strength saturates at (-strength) */
//         strength_x = -strength/2*(1+tanh(5*(strength_x+13.82)/2.30));
//         strength_y = -strength/2*(1+tanh(5*(strength_y+13.82)/2.30));
//         strength_z = -strength/2*(1+tanh(5*(strength_z+13.82)/2.30));


         if (0&&master()) {
            fprintf(stderr,"Computed iLES filter strengths: %g %g %g\n",strength_x,strength_y,strength_z);
         }

         /* Rescale the k,l,m vectors such that the filter is of the appropriate
            strength at the cutoff */
         kvec = exp(-strength_x*pow(abs(kvec),order)/pow(kcut,order));
         lvec = exp(-strength_y*pow(abs(lvec),order)/pow(lcut,order));
         mvec = exp(-strength_z*pow(abs(mvec),order)/pow(mcut,order));

         

//         exit(1);
      
   }
      
   if (!implicit_les) {
      /* If we aren't using an iles filter, we want one where low wavenumbers
         are completely unchanged; after the cutoff value the filter
         behaves as an (order)-order Gaussian, decaying to exp(-cutoff) at
         the Nyquist frequency */
   kvec = where(abs(kvec) <= cutoff*kmax, 1, exp(-strength*pow((kvec-kcut)/(kmax-kcut), order)));
   lvec = where(abs(lvec) <= cutoff*lmax, 1, exp(-strength*pow((lvec-lcut)/(lmax-lcut), order)));
   mvec = where(abs(mvec) <= cutoff*mmax, 1, exp(-strength*pow((mvec-mcut)/(mmax-mcut), order)));
   }

   // Multiply the proper temporary array by the transformation coefficients

   if (tform.use_complex()) {
      CTArray & temp = *(tform.get_complex_temp());
      temp = temp * kvec(ii) * lvec(jj) * mvec(kk) / norm;
   } else {
      DTArray & temp = *(tform.get_real_temp());
      temp = temp * kvec(ii) * lvec(jj) * mvec(kk) / norm;
   }

   // Transform back to overwrite source.
   tform.back_transform(&source, dim1_type, dim2_type, dim3_type);
}
            
void write_array(Array<double, 3> const & ar_d, const string basename, int seq_num, MPI_Comm c) {
   ofstream file;
   /* Cast the array to single precision floating point for smaller filesize*/
   static int myrank=-1, numproc=-1;
   static MPI_Comm last_comm = MPI_COMM_WORLD;
   if (myrank < 0 || numproc < 0 || last_comm != c) {
      /* Find out where we are in the communicator */
      last_comm = c;
      MPI_Comm_rank(last_comm,&myrank);
      MPI_Comm_size(last_comm,&numproc);
   }
   GeneralArrayStorage<3> output_store;
   /* Create a storage order to put the split dimension last in memory, so that
      appending to a file works */
//   if (numproc > 1) { // parallel case
   /* Tweak -- use the "parallel" output ordering for all arrays, even when
      writing in the single-processor case.  This allows full flexibility
      on reading in for restart files without a need for ordering-metadata */
      output_store.ordering()[0] = secondDim;
      output_store.ordering()[1] = thirdDim;
      output_store.ordering()[2] = firstDim;
//   } else {
//      output_store.ordering()[ar_d.ordering(firstDim)] = firstDim;
//      output_store.ordering()[ar_d.ordering(secondDim)] = secondDim;
//      output_store.ordering()[ar_d.ordering(thirdDim)] = thirdDim;
//   }
//   Array<float, 3> ar(ar_d.lbound(),ar_d.extent(),output_store); 
//   ar = cast<float>(ar_d);
   Array<double, 3> ar(ar_d.lbound(),ar_d.extent(),output_store); 
   ar = ar_d;
   string filename;
   if (seq_num >= 0) { /* "Sequenced" output */
      std::stringstream conv; /* For converting sequence number to string */
      conv << basename;
      conv << ".";
      conv << seq_num;
      filename = conv.str();
   } else {
      filename = basename;
   }
   /* Now, loop over each process, and append our data to the file in turn.  This
      serializes data output, but results in a single file.  MPI-IO would probably
      be a performance improvement, and NetCDF-4 would be a self-documenting extension. */
   for (int i = 0; i < numproc; i++) {
      if (myrank != i) { // Don't write
         MPI_Barrier(last_comm);
      } else { // do write
         // Append to the end of the file
         
         if (master(last_comm)) // If we're processor 0, overwrite the file contents
            file.open(filename.c_str(),std::ios::binary | std::ios::trunc);
         else // else append to the end
            file.open(filename.c_str(),std::ios::binary | std::ios::ate | std::ios::app);
         if (!ar.isStorageContiguous()) {
            /* For now, do not support noncontiguous arrays */
            abort();
         }
         /* Write the array to file in memory order */
         file.write((char *) ar.data(), sizeof(double)*ar.numElements());
         file.close();
         MPI_Barrier(last_comm);
      }
   }
}

void write_reader(Array<double, 3> const & ar, const string basename, bool seq, MPI_Comm c) {
   /* Creates a MATLAB .m file designed to read this program's output into
      a sensible 3D array */
   int sizes[3] = {0,0,0};
   /* First, accumulate global sizes, assuming that the array is split over the x-dimension*/
   int local_x = ar.extent(firstDim);
   sizes[0] = pssum(local_x,c);
   sizes[1] = ar.extent(secondDim);
   sizes[2] = ar.extent(thirdDim);
   if (!master(c)) return; // only master process writes the m-file
   int numproc = 0;
   MPI_Comm_size(c,&numproc);
   int orderings[3];
   /* Single processor and multiprocessor runs have different orderings, and
      the ordering written to disk is different (generally) than that in memory */
   /* Update -- as with change in write_array, single and multiprocessor runs will
      have the same ordering, optimized for the parallel case. */
//   if (numproc > 1) {
      orderings[0] = secondDim;
      orderings[1] = thirdDim;
      orderings[2] = firstDim;
//   } else {
//      orderings[ar.ordering(firstDim)] = firstDim;
//      orderings[ar.ordering(secondDim)] = secondDim;
//      orderings[ar.ordering(thirdDim)] = thirdDim;
//   }
   char* endianess; // Endianess check
   {
      char EndianCheck[2] = {1, 0};
      short x = *(reinterpret_cast<short *>(EndianCheck));
      if (x ==  1) { // Little endian system
         endianess = "L";
      } else if (x == 256) { // Big endian system
         endianess = "B";
      } else { // Unknown endianess?
         abort();
      }
   }
   string filename = basename + "_reader.m";
   ofstream file(filename.c_str());
   // Write out a .m-file that permits slab-reading; this code is originally 
   // courtesy of Michael Dunphy, mdunphy@uwaterloo.ca

   file << "function ar = " <<basename << "_reader(";
   if (seq) file << "seq,";
   file << "xrange,";
   if (sizes[1] > 1) file << "yrange,";
   file << "zrange)\n";
   file << "%" << basename << "_reader - SPINS data reader with slab support\n";
   file << "% Opens and reads a slab of SPINS data, optionally\n";
   file << "% loading only a portion of the total array. This\n";
   file << "% functionality is most useful when only a portion\n";
   file << "%  of a large, 3D array is needed.\n";
   file << "%\n% Usage:\n";
   file << "%   slab = " << basename << "_reader(";
   if (seq) file << "seq,";
   file << "xrange,";
   if (sizes[1] > 1) file << "yrange,";
   file << "zrange);\n";
   file << "% Where ";
   if (seq) file << "seq is the sequence number of the output and\n% ";
   file << "xrange";
   if (sizes[1] > 1) file << ", yrange,";
   file << " and zrange are the ranges\n% of the array to be read.\n";
   file << "% Empty values, [], and : each imply reading the\n% full dimension.\n\n";

   file << "% Version 1.1, July 09 2012.  Original general\n";
   file << "% MATLAB code provided courtesy of Michael Dunphy\n";
   file << "% (mdunphy@uwaterloo.ca), adapted for SPINS by \n";
   file << "% Christopher Subich (csubich@uwaterloo.ca).\n\n";

   // Write out argument-sanitizers
   file << "% Sanitize the ranges:\n";
   file << "if (~exist('xrange') || isempty(xrange) || isequal(xrange,':')) xrange = [1:" << sizes[0] << "]; end;\n";
   if (sizes[1] > 1) 
      file << "if (~exist('yrange') || isempty(yrange) || isequal(yrange,':')) yrange = [1:" << sizes[1] << "]; end;\n";
   else
      file << "yrange = [1];\n";
   file << "if (~exist('zrange') || isempty(zrange) || isequal(zrange,':')) zrange = [1:" << sizes[2] << "]; end;\n";

   file << "xrange(xrange < 1) = []; xrange(xrange > " << sizes[0] << ") = [];\n";
   file << "yrange(yrange < 1) = []; yrange(yrange > " << sizes[1] << ") = [];\n";
   file << "zrange(zrange < 1) = []; zrange(zrange > " << sizes[2] << ") = [];\n";

   file << "\n% Define ranges in file-ordering\n";
   file << "ranges = {xrange,yrange,zrange};\n";
   file << "ranges = ranges([" <<
      1+orderings[firstDim] << "," <<
      1+orderings[secondDim] << "," <<
      1+orderings[thirdDim] << "]);\n";
   
   file << "% Output file name:\n";
   file << "fname = ";
   if (seq)
      file << "sprintf('%s.%d','" << basename << "',seq);\n";
   else
      file << "'" << basename << "';\n";

   file << "\n% memmap the file for reading\n";
   file << "m = memmapfile(fname, 'Offset',0, ...\n";
   file << "   'Format', {'double' [";
   file << sizes[orderings[firstDim]] << ","
        << sizes[orderings[secondDim]] << ","
        << sizes[orderings[thirdDim]] << "]";
   file << " 'x'}, ...\n";
   file << "   'Writable',false);\n";

   file << "\n% Extract the data and clear the memmap\n";
   file << "ar = m.Data.x(ranges{1},ranges{2},ranges{3}); clear m\n";

   file << "\n% Permute, check endianness, and return\n";
   file << "ar = squeeze(ipermute(ar,[" <<
      1+orderings[firstDim] << "," <<
      1+orderings[secondDim] << "," <<
      1+orderings[thirdDim] << "]));\n";
   file << "[~,~,endian] = computer();\n";
   file << "if (~isequal(endian,'" << endianess << "'))\n";
   file << "   ar = swapbytes(ar);\n";
   file << "end\n";

#if 0
   file <<
      "function ar = " << basename << "_reader(";
   if (seq) /* If this is sequenced output, take the sequence number as param*/
      file << "seq";
   file << ")\n";
   file << "  afile = fopen(";
   if (seq)
      file << "sprintf('%s.%d','" << basename << "',seq)";
   else
      file << "'" << basename << "'";
   file << ",'r');\n";
   file << "  ar = fread(afile,inf,'double',";
   /* Write endian specifier into the fread command */
   file << "'" << endianess << "'" << ");\n";
   /* Now, ar is a 1-by-many vector containing all the written values.  We want
      to reshape it into a 3D array, in storage order. */
   file << "  ar = reshape(ar,[";
   file << sizes[orderings[firstDim]] << ","
        << sizes[orderings[secondDim]] << ","
        << sizes[orderings[thirdDim]] << "]);\n";
   /* Permute the array into algebraic order */
   file << "  % Permute the array into logical order, rather than memory order\n";
   file << "  ar = squeeze(ipermute(ar,["
        << orderings[firstDim]+1 << ","
        << orderings[secondDim]+1 << ","
        << orderings[thirdDim]+1 << "]));\n";
   /* Close the file pointer */
   file << "  fclose(afile);\n";
//   file << "  [Ctype MMax Endian] = computer();\n";
#endif
   file.close();
}

void read_array(blitz::Array<double,3> & ar, const char * filename,
      int size_x, int size_y, int size_z, MPI_Comm c) {
   /* Read from an on-disk array of size_x by size_y by size_z to the local
      array ar, respecting that ar will probably be a subset of the full
      array.  Required backend for reading restart files */
   GeneralArrayStorage<3> file_order; // Array storage order for on-file array
   /* Copied from write_array, above */
   file_order.ordering()[0] = secondDim;
   file_order.ordering()[1] = thirdDim;
   file_order.ordering()[2] = firstDim;

   int myrank, numproc;
   MPI_Comm_rank(c,&myrank);
   MPI_Comm_size(c,&numproc);

   for (int i = 0; i < numproc; i++) {
      if (i != myrank) {
         /* Synchronize with other processors so only one maps the file
            at a time */
         MPI_Barrier(c);
         continue;
      }

      fprintf(stderr,"Processor %d mapping file %s\n",myrank,filename);
      
      int my_fd = open(filename,O_RDONLY); // Open the file for mapping
      if (my_fd == -1) {
         fprintf(stderr,"I/O error opening %s for reading\n",filename);
         fprintf(stderr,"The error was: %d, %s\n",errno,strerror(errno));
         exit(1);
      }

      /* Map the array into memory, given that it is of size_x * size_y * size_z.
         If the array is too amazingly big, this -might- require recompilation with
         memmodel=medium to use larger data pointers. */
      double * disk_data = (double *) mmap(0,sizeof(double)*size_x*size_y*size_z,
                                          PROT_READ,MAP_SHARED,my_fd,0);
      if (disk_data == MAP_FAILED) {
         fprintf(stderr,"Memmap of %s failed\n",filename);
         fprintf(stderr,"The error was: %d, %s\n",errno,strerror(errno));
         exit(1);
      }

      /* Create a blitz array that references the mmap'd array on disk.  This
         will allow read-only access to the array using convenient Blitz notation */
      blitz::Array<double,3> disk_array(disk_data,blitz::shape(size_x,size_y,size_z),
                                          blitz::neverDeleteData,file_order);

      /* Assign using Blitz's domain syntax */
      ar(ar.domain()) = disk_array(ar.domain());

      /* And unmap the memory */
      if (munmap(disk_data,sizeof(double)*size_x*size_y*size_z) == -1) {
         fprintf(stderr,"Memory unmap of %s failed\n",filename);
         fprintf(stderr,"The error was: %d, %s\n",errno,strerror(errno));
         exit(1);
      }
      /* Synchronize with the other processors */
      MPI_Barrier(c);
   }
}

} // End namespace

