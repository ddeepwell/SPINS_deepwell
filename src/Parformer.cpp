/* Parformer.cpp -- implementation of parallel replacement for TransWrapper. */

#include "Parformer.hpp"
#include "Par_util.hpp"
#include <blitz/tinyvec-et.h>
#include <stdio.h>
#include <iostream>

using namespace std;

namespace Transformer {

//   namespace {
      static bool isreal(S_EXP t) {
         return (t == REAL || t == SINE || t == COSINE || t == CHEBY);
      }
      static bool iscomplex(S_EXP t) {
         return (t == COMPLEX || t == FOURIER);
      }

//   }
   /* Constructor */
   TransWrapper::TransWrapper(int szx, int szy, int szz,
         S_EXP Ttx, S_EXP Tty, S_EXP Ttz,
         MPI_Comm c):
      Tx(Ttx), Ty(Tty), Tz(Ttz), sx(szx), sy(szy), sz(szz),
      realTransposer(0), complexTransposer(0),
      yz_real_temp(0), x_real_temp(0), yz_compl_temp(0), x_compl_temp(0) {

         //fprintf(stderr,"Initializing TransWrapper\n");
         /* Set the reference counter */
         refcount = new int(1);        
         communicator = c;

         /* If one of the input array sizes is 1, then there really isn't
            a sensible transform */
         if (sx == 1) Tx = NONE;
         if (sy == 1) Ty = NONE;
         if (sz == 1) Tz = NONE;

         /* The real transposer will be used to allocate any real temporaries */
         realTransposer = new Transposer<double> (sx, sy, sz,
                                          firstDim, thirdDim, c);

         /* If there's a null transform, the the TransWrapper reduces
            to a simple "copy to temporary" */
         if (Tx == NONE && Ty == NONE && Tz == NONE) {
            x_real_temp = alloc_array(szx,szy,szz,c);
//            x_real_temp = new DTArray(x_real_lbound, x_real_extent,
//                                    x_real_storage);
            return;
         }
            

         /* Now, to figure out what temporaries we're going to need.
            First, we'll transform one of the y/z dimensions, preferring
            to go real then complex (if both are there).  If either is real,
            therefore, we'll need the real temporary */
         /* Calculate lbound/extent regardless, because we'll use this
            as a basis for the complex y/z temporary if necessary */
         realTransposer->source_alloc(yz_real_lbound,yz_real_extent,
                              yz_real_storage);
         if (isreal(Ty) || isreal(Tz)) {
            //fprintf(stderr,"Initializing yz_real_temp\n");
            yz_real_temp = new DTArray (yz_real_lbound, yz_real_extent,
                                    yz_real_storage);
         }
         /* Now, generate the complex temporaries if necessary */
         if (iscomplex(Ty)) {
            yz_compl_extent =  yz_real_extent;
            yz_compl_lbound = yz_real_lbound;
            yz_compl_storage = yz_real_storage;
            /* Real-to-complex transforms change array sizes */
            yz_compl_extent(secondDim) = yz_compl_extent(secondDim)/2 + 1;
            //fprintf(stderr,"Initializing yz_compl_temp\n");
            yz_compl_temp = new CTArray (yz_compl_lbound, yz_compl_extent, 
                                    yz_compl_storage);
         } else if (iscomplex(Tz)) {
            yz_compl_extent =  yz_real_extent;
            yz_compl_lbound = yz_real_lbound;
            yz_compl_storage = yz_real_storage;
            yz_compl_extent(thirdDim) = yz_compl_extent(thirdDim)/2 + 1;
            //fprintf(stderr,"Initializing yz_compl_temp\n");
            yz_compl_temp = new CTArray (yz_compl_lbound, yz_compl_extent, 
                                    yz_compl_storage);
         }
         /* Now, to allocate the x-temporary */
         /* Assert we don't do a real transform after a complex one.  This
            is an essentially arbitrary limitation (since CTArray doesn't
            support real transforms) that should be lifted for full
            generality (solid wall depth, streamwise, periodic spanwise) */
         if (isreal(Tx)) assert(!yz_compl_temp);

         if (!yz_compl_temp && Tx != NONE) {
            /* We have to transpose to make the x-dimension contiguous.  If
               we haven't done any complex transforms yet (so we're still
               dealing with real values), we'll want to use the realTransposer
               to allocate the destination temporary */
            realTransposer->dest_alloc(x_real_lbound, x_real_extent,
                                 x_real_storage);
            //fprintf(stderr,"Initializing x_real_temp\n");
            x_real_temp = new DTArray(x_real_lbound, x_real_extent,
                                    x_real_storage);
            if (iscomplex(Tx)) {
               /* Further, we'll also need to allocate the complex temporary
                  if Tx is itself a complex transform.  The size along
                  the x dimension will change becaue of the real-to-complex
                  transform. */
               x_compl_lbound = x_real_lbound;
               x_compl_extent = x_real_extent;
               x_compl_storage = x_real_storage;

               x_compl_extent(firstDim) = x_compl_extent(firstDim)/2 + 1;
//            //fprintf(stderr,"Initializing x_compl_temp\n");
//            cerr << x_compl_lbound << endl;
//            cerr << x_compl_extent << endl;
//            cerr << x_compl_storage.ordering() << endl;
               x_compl_temp = new CTArray(x_compl_lbound, x_compl_extent,
                                       x_compl_storage);
//               //fprintf(stderr,"Initialized\n");
            }
         }
         if (yz_compl_temp && Tx != NONE) {
            /* We'll be transposing a complex array, rather than a real one.
               Create and use complexTransposer */
            complexTransposer = new Transposer<complex<double> >
                              (sx, yz_compl_extent(secondDim), 
                               yz_compl_extent(thirdDim),
                               firstDim, thirdDim, c);
            complexTransposer->dest_alloc(x_compl_lbound, x_compl_extent,
                                       x_compl_storage);
//            //fprintf(stderr,"Initializing x_compl_temp\n");
            x_compl_temp = new CTArray(x_compl_lbound, x_compl_extent,
                                    x_compl_storage);
         }
/*         fprintf(stderr,"Transformer debugging information:\n");
         fprintf(stderr,"x_real: %d, y_real: %d, z_real: %d\n",
               isreal(Tx),isreal(Ty),isreal(Tz));
         fprintf(stderr,"x_compl: %d, y_compl: %d, z_real: %d\n",
               iscomplex(Tx),iscomplex(Ty),iscomplex(Tz));
         fprintf(stderr,"x_real_temp: %d, x_compl_temp: %d\n",
               x_real_temp != 0, x_compl_temp != 0);
         fprintf(stderr,"yz_real_temp: %d, yz_compl_temp: %d\n",
               yz_real_temp != 0, yz_compl_temp != 0);*/
      }

   /* Copy constructor */
   TransWrapper::TransWrapper(const TransWrapper & copyfrom) {
      Tx = copyfrom.Tx; Ty = copyfrom.Ty; Tz = copyfrom.Tz;
      sx = copyfrom.sx; sy = copyfrom.sy; sz = copyfrom.sz;

      realTransposer = copyfrom.realTransposer;
      complexTransposer = copyfrom.complexTransposer;
      
      /* Increment reference counter for temporary arrays */
      refcount = copyfrom.refcount; (*refcount)++;
      yz_real_temp = copyfrom.yz_real_temp;
      x_real_temp = copyfrom.x_real_temp;
      yz_compl_temp = copyfrom.yz_compl_temp;
      x_compl_temp = copyfrom.x_compl_temp;

      yz_real_lbound = copyfrom.yz_real_lbound;
      yz_real_extent = copyfrom.yz_real_extent;
      x_real_lbound = copyfrom.x_real_lbound;
      x_real_extent = copyfrom.x_real_extent;
      x_compl_lbound = copyfrom.x_compl_lbound;
      x_compl_extent = copyfrom.x_compl_extent;
      yz_compl_lbound = copyfrom.yz_compl_lbound;
      yz_compl_extent = copyfrom.yz_compl_extent;

      yz_real_storage = copyfrom.yz_real_storage;
      yz_compl_storage = copyfrom.yz_compl_storage;
      x_real_storage = copyfrom.x_real_storage;
      x_compl_storage = copyfrom.x_compl_storage;
/*         fprintf(stderr,"Transformer (copy) debugging information:\n");
         fprintf(stderr,"x_real: %d, y_real: %d, z_real: %d\n",
               isreal(Tx),isreal(Ty),isreal(Tz));
         fprintf(stderr,"x_compl: %d, y_compl: %d, z_real: %d\n",
               iscomplex(Tx),iscomplex(Ty),iscomplex(Tz));
         fprintf(stderr,"x_real_temp: %d, x_compl_temp: %d\n",
               x_real_temp != 0, x_compl_temp != 0);
         fprintf(stderr,"yz_real_temp: %d, yz_compl_temp: %d\n",
               yz_real_temp != 0, yz_compl_temp != 0);*/
   }

   /* Destructor */
   TransWrapper::~TransWrapper() {
      (*refcount)--; // decrement reference count
      if ((*refcount) <= 0) {
         assert (*refcount == 0); // Sanity check
         /* If refcount is 0, we were the last one using this stuff */
         delete refcount;
         if (realTransposer) delete realTransposer;
         if (complexTransposer) delete complexTransposer;
         if (yz_real_temp) delete yz_real_temp;
         if (x_real_temp) delete x_real_temp;
         if (yz_compl_temp) delete yz_compl_temp;
         if (x_compl_temp) delete x_compl_temp;
      }
   }

   bool TransWrapper::use_complex()  const {
      if (iscomplex(Tx)) return true;
      if (isreal(Tx)) return false;
      if (iscomplex(Ty) || iscomplex (Tz)) return true;
      return false;
   }

   CTArray * TransWrapper::get_complex_temp()  const {
      /* Returns the pointer to the last-used complex temporary */
      if (iscomplex(Tx)) {
         /* x-transform was complex, so we did it last */
         assert(x_compl_temp);
         return x_compl_temp;
      } 
      /* If x isn't complex, to return a complex temporary then y or z
         must be */
      assert(iscomplex(Ty) || iscomplex(Tz));
      assert(yz_compl_temp);
      return yz_compl_temp;
   }

   DTArray * TransWrapper::get_real_temp() const {
      /* Returns the pointer to the last-used temporary if real */
      if (isreal(Tx)) {
         assert(x_real_temp);
         return x_real_temp;
      }
      assert( !iscomplex(Ty) && !iscomplex(Tz));
      assert(yz_real_temp);
      return yz_real_temp;
   }

   double TransWrapper::norm_factor() const {
      double factor = 1;
      if (isreal(Tx)) {
         if (Tx == CHEBY)
            factor *= 2*(sx-1);
         else
            factor *= 2*sx;
      } else if (iscomplex(Tx)) {
         factor *= sx;
      } if (isreal(Ty)) {
         if (Ty == CHEBY)
            factor *= 2*(sy-1);
         else
            factor *= 2*sy;
      } else if (iscomplex(Ty)) {
         factor *= sy;
      } if (isreal(Tz)) {
         if (Tz == CHEBY)
            factor *= 2*(sz-1);
         else
            factor *= 2*sz;
      } else if (iscomplex(Tz)) {
         factor *= sz;
      }
      return factor;
   }

   double TransWrapper::max_wavenum(Dimension dim) {
      /* Returns the maximum wavenumber along a dimension dim, useful
         for filtering and such when the entire wavenumber array might
         not be present */
      S_EXP type;
      int size;
      if (dim == firstDim) {type = Tx; size = sx;}
      else if (dim == secondDim) {type = Ty; size = sy;}
      else if (dim == thirdDim) {type = Tz; size = sz;}
      else assert(dim == firstDim || dim == secondDim || dim == thirdDim);
      assert(size);
      switch(type) {
         case NONE:
            return 0;
         case SINE:
            return size; // no zero-freuqency, so maximum frequency is size
         case COSINE: case CHEBY:
            /* In the cosine case, this refers to k in a_k cos(k*x).  In the
               Cehbyshev case, this refers to which T_k polynomial the index
               refers to */
            return size-1; // Either way, there's a 0-frequency
         case FOURIER:
            /* A Fourier dimension has both positive and negative frequencies,
               and the positives only run up to (size+1)/2 
               (size 1 -- DC only.  Size 2 -- DC and Nyquist)*/
            return double(int((size+1)/2));
         default:
            abort(); // No specific transform made yet.  Oopsie!
      }
   }
      
   Array<double,1> TransWrapper::wavenums(Dimension dim,int lb,int ub)  const {
      /* Returns wavenumbers along a given dimension, in transposed space,
         based on the proper temporary array */
      blitz::firstIndex ii;
      if ((lb == 0 && ub == 0)) {
         if (use_complex()) {
            lb = get_complex_temp()->lbound(dim);
            ub = get_complex_temp()->ubound(dim);
         } else { // if (Tx != NONE || Ty != NONE || Tz != NONE) {
            lb = get_real_temp()->lbound(dim);
            ub = get_real_temp()->ubound(dim);
         } 
      }
      assert(ub >= lb); // make sure we have a nonzero-length vector
      blitz::Range r(lb,ub);
      Array<double,1> out(r);
      out = 0;

      S_EXP type;
      int size;
      if (dim == firstDim) {type = Tx; size = sx;}
      else if (dim == secondDim) {type = Ty; size = sy;}
      else if (dim == thirdDim) {type = Tz; size = sz;}
      else assert(dim == firstDim || dim == secondDim || dim == thirdDim);
      assert(size);

      switch(type) {
         case NONE:
            out = 0;
            break;
         case SINE:
            for (int i = lb; i <= ub; i++) {
               out(i) = i+1; // No zero-frequency in a sine transform
            }
            break;
         case COSINE: case CHEBY:
            /* In the cosine case, this refers to k in a_k cos(k*x).  In the
               Cehbyshev case, this refers to which T_k polynomial the index
               refers to */
            for (int i = lb; i <= ub; i++) {
               out(i) = i;
            }
            break;
         case FOURIER:
            /* A somewhat complicated case, since the Fourier case has both
               positive and negative frequencies.  Fortunately, a "where"
               statement allows proper handling: */
            out = where(ii <= size/2, ii, (ii-size));
            break;
         default:
            abort(); // No specific transform made yet.  Oopsie!
      }
               
      return out;
   }

   /* Do the forward transform */
   void TransWrapper::forward_transform(DTArray * in, S_EXP Ttx,
         S_EXP Tty, S_EXP Ttz) {
      /* If one of the input array sizes is 1, then there really isn't
         a sensible transform */
//      fprintf(stderr,"TransWrapper forward transform\n");
//      fprintf(stderr,"x: %d/%d, y: %d/%d, z: %d/%d\n",Ttx,Tx,Tty,Ty,Ttz,Tz);
      if (sx == 1) Ttx = NONE;
      if (sy == 1) Tty = NONE;
      if (sz == 1) Ttz = NONE;

      /* If there's a null transform, copy to the temporary and return */
      if (Ttx == NONE && Tty == NONE && Ttz == NONE) {
         *x_real_temp = *in;
         return;
      }

      /* Now, transform real y, real z, complex y, complex z -- only at
         most 2 will apply */

      DTArray * src = in; // Sourced array
      ::TArrayn::Trans t_type; // TArray-specific transform type

      //fprintf(stderr,"yz real transforms\n");
      if (isreal(Ty)) {
         assert (isreal(Tty));
         Ty = Tty;
         //fprintf(stderr,"y real->real\n");
         /* Find the low-level transform type */
         if (Tty == SINE) t_type = ::TArrayn::DST1;
         else if (Tty == COSINE) t_type = ::TArrayn::DCT1;
         else { assert(Tty == CHEBY); t_type = ::TArrayn::DCT0; }

         src->transform(*yz_real_temp,secondDim,t_type);
         src = yz_real_temp;
      }
      if (isreal(Tz)) {
         assert (isreal(Ttz));
         Tz = Ttz;
         //fprintf(stderr,"z real->real\n");
         if (Ttz == SINE) t_type = ::TArrayn::DST1;
         else if (Ttz == COSINE) t_type = ::TArrayn::DCT1;
         else { assert(Ttz == CHEBY); t_type = ::TArrayn::DCT0; }

         src->transform(*yz_real_temp,thirdDim,t_type);
         src = yz_real_temp;
      }

      /* Now, do y/z complex transforms */
      //fprintf(stderr,"yz complex transforms\n");
      bool is_complex = false;
      CTArray * c_src = 0;

      if (iscomplex(Ty)) {
         assert(iscomplex(Tty));
         Ty = Tty;
         assert(Tty == FOURIER);
         if (is_complex) { // previous transform was complex
            //fprintf(stderr,"y complex->complex\n");
           t_type = ::TArrayn::FFT;
           c_src->transform(*yz_compl_temp,secondDim,t_type);
           c_src = yz_compl_temp;
         } else  { // previous transform was real 
            //fprintf(stderr,"y real->complex\n");
            t_type = ::TArrayn::FFTR;
            src->transform(*yz_compl_temp,secondDim,t_type);
            c_src = yz_compl_temp;
            is_complex = true;
         }
      }

      if (iscomplex(Tz)) {
         assert(iscomplex(Ttz));
         Tz = Ttz;
         assert(Ttz == FOURIER);
         if (is_complex) { // previous transform was complex
            //fprintf(stderr,"z complex->complex\n");
            t_type = ::TArrayn::FFT;
            c_src->transform(*yz_compl_temp,thirdDim,t_type);
            c_src = yz_compl_temp;
         } else { // Previous transform was real
            //fprintf(stderr,"z real->complex\n");
            t_type = ::TArrayn::FFTR;
            src->transform(*yz_compl_temp,thirdDim,t_type);
            c_src = yz_compl_temp;
            is_complex = true;
         }
      } 

      /* Handle real x, then complex x */

      //fprintf(stderr,"real x\n");
      if (isreal(Tx)) {
         /* We don't currently support a real transform after a complex one */
         assert (!is_complex); 
         /* From the real temporary, transpose to x_real_temp */
         realTransposer->transpose(*src,*x_real_temp);
         
         /* Now, transform in place */
         Tx = Ttx;
         if (Tx == SINE) t_type = ::TArrayn::DST1;
         else if (Tx == COSINE) t_type = ::TArrayn::DCT1;
         else {assert(Tx == CHEBY); t_type = ::TArrayn::DCT0;}
         //fprintf(stderr,"x real->real\n");
         x_real_temp->transform(*x_real_temp,firstDim,t_type);
      }
      //fprintf(stderr,"x complex\n");
      if (iscomplex(Tx)) {
         if (is_complex) {
            /* Complex all the way dowm, so transpose yz_compl_temp */
            //fprintf(stderr,"x complex->complex\n");
            complexTransposer->transpose(*yz_compl_temp,*x_compl_temp);
            Tx = Ttx;
            assert(Tx == FOURIER); t_type = ::TArrayn::FFT;
/*            cerr << yz_compl_temp->lbound() << endl << 
               yz_compl_temp->ubound() << endl << 
               x_compl_temp->lbound() << endl <<
               x_compl_temp->ubound() << endl;*/
            x_compl_temp->transform(*x_compl_temp,firstDim,t_type);
         } else {
            /* real-to-complex transform */
            //fprintf(stderr,"x real->complex (%g)\n",pvmax(*src));
            realTransposer->transpose(*src,*x_real_temp);
            //fprintf(stderr,"!!%g!!\n",pvmax(*x_real_temp));
            //fprintf(stderr,"!!%g!!\n",psmax(max(abs(*x_compl_temp))));
            Tx = Ttx;
            assert(Tx == FOURIER); t_type = ::TArrayn::FFTR;
            x_real_temp->transform(*x_compl_temp,firstDim,t_type);
            //fprintf(stderr,"!!%g!!\n",psmax(max(abs(*x_compl_temp))));
         }
      }
   }

   void TransWrapper::back_transform(Array<double,3> * out, S_EXP Ttx,
         S_EXP Tty, S_EXP Ttz) {
      /* If one of the input array sizes is 1, then there really isn't
         a sensible transform */
      if (sx == 1) Ttx = NONE;
      if (sy == 1) Tty = NONE;
      if (sz == 1) Ttz = NONE;
//      fprintf(stderr,"x: %d/%d, y: %d/%d, z: %d/%d\n",Ttx,Tx,Tty,Ty,Ttz,Tz);

      /* Inverse transform goes in the reverse order of the forward transform
         above.  First, complex/real x */

      /* It helps to know whether yz transforms collectively have complex 
         output, since that tells us if x should end up transposing into 
         yz_compl or yz_real */

      /* If there's a null transform, copy back from the temporary */
      if (Ttx == NONE && Tty == NONE && Ttz == NONE) {
         *out = *x_real_temp;
         return;
      }

      bool yz_compl = iscomplex(Tty) || iscomplex(Ttz);
      bool yz_none = (Tty == NONE && Ttz == NONE);

      ::TArrayn::Trans t_type; // Lower-level transform-type
      //fprintf(stderr,"x reverse transform\n");
      if (iscomplex(Ttx)) {
         assert(iscomplex(Tx));
         Tx = Ttx; assert(Ttx == FOURIER);
         if (yz_compl) {
            /* Complex to complex transform*/
            //fprintf(stderr,"x reverse complex->complex\n");
            t_type = ::TArrayn::IFFT;
            x_compl_temp->transform(*x_compl_temp,firstDim,t_type);
            complexTransposer->back_transpose(*x_compl_temp,*yz_compl_temp);
         } else {
            /* Complex to real transform */
            //fprintf(stderr,"x reverse comlex->real\n");
            t_type = ::TArrayn::IFFTR;
            x_compl_temp->transform(*x_real_temp,firstDim,t_type);
            Array<double,3> * final_out;
            if (yz_none) final_out = out;
            else final_out = yz_real_temp;
            realTransposer->back_transpose(*x_real_temp,*final_out);
         }
      }
      if (isreal(Ttx)) {
         //fprintf(stderr,"x reverse real->real\n");
         assert(isreal(Tx));
         assert(!yz_compl);
         Tx = Ttx;
         switch (Ttx) {
            case SINE: t_type = ::TArrayn::IDST1; break;
            case COSINE: t_type = ::TArrayn::IDCT1; break;
            case CHEBY: t_type = ::TArrayn::DCT0; break;
            default: abort();
         }

         x_real_temp->transform(*x_real_temp,firstDim,t_type);
         /* Now, find out the transpose destination */
         Array<double,3> * final_out;
         if (yz_none) final_out = out;
         else final_out = yz_real_temp;
         realTransposer->back_transpose(*x_real_temp,*final_out);
      }

      /* Now, y/z transforms.  Since we transformed in yreal/zreal/ycompl/zcompl
         order, we have to reverse that here.  */
      //fprintf(stderr,"yz reverse transforms\n");
      if (iscomplex(Ttz)) {
         assert(iscomplex(Tz) && Ttz == FOURIER);
         Tz = Ttz;
         if (!iscomplex(Tty)) {
            /* The other transform is either real or none */
            //fprintf(stderr,"z reverse complex->real\n");
            t_type = ::TArrayn::IFFTR;
            Array<double,3> * final_out;
            if (Tty == NONE) final_out = out;
            else final_out = yz_real_temp;
            yz_compl_temp->transform(*final_out,thirdDim,t_type);
         } else {
            //fprintf(stderr,"z reverse complex->complex\n");
            t_type = ::TArrayn::IFFT;
            yz_compl_temp->transform(*yz_compl_temp,thirdDim,t_type);
         }
      }
      if (iscomplex(Tty)) {
         //fprintf(stderr,"y reverse complex->real\n");
         assert(iscomplex(Ty) && Tty == FOURIER);
         Ty = Tty;
         t_type = ::TArrayn::IFFTR;
         Array<double,3> * final_out;
         if (Ttz == NONE || iscomplex(Ttz)) final_out = out;
         else final_out = yz_real_temp;
         yz_compl_temp->transform(*final_out,secondDim,t_type);
      }
      if (isreal(Ttz)) {
         assert(isreal(Tz));
         Tz = Ttz;
         //fprintf(stderr,"z reverse real->real\n");
         switch(Ttz) {
            case SINE: t_type = ::TArrayn::IDST1; break;
            case COSINE: t_type = ::TArrayn::IDCT1; break;
            case CHEBY: t_type = ::TArrayn::DCT0; break;
            default: abort();
         }
         Array<double,3> * final_out;
         if (Tty == NONE || iscomplex(Tty)) final_out = out;
         else final_out = yz_real_temp;
//         assert (!any(blitz_isnan(*yz_real_temp)));
         yz_real_temp->transform(*final_out,thirdDim,t_type);
//         assert (!any(blitz_isnan(*final_out)));
      }
      if (isreal(Tty)) {
         assert(isreal(Ty));
         Ty = Tty;
         //fprintf(stderr,"y reverse real->real\n");
         switch(Tty) {
            case SINE: t_type = ::TArrayn::IDST1; break;
            case COSINE: t_type = ::TArrayn::IDCT1; break;
            case CHEBY: t_type = ::TArrayn::DCT0; break;
            default: abort();
         }
         /* This is the last possible transform, so we know we're going
            to the specified output */
         yz_real_temp->transform(*out,secondDim,t_type);
      }
//      assert (!any(blitz_isnan(*out)));
   }

   Trans1D::Trans1D(int szx, int szy, int szz, Dimension dim, S_EXP type,
         MPI_Comm c):
      TransWrapper(szx, szy, szz, // sizes
            (dim == firstDim ? type : NONE), // Expand full constructor
            (dim == secondDim ? type : NONE),
            (dim == thirdDim ? type : NONE),c) {
         trans_dim = dim;
      }
   Trans1D::Trans1D(const Trans1D & copyfrom): // copy constructor
      TransWrapper(copyfrom), trans_dim(copyfrom.trans_dim) {
      }

   Dimension Trans1D::getdim() {
      return trans_dim;
   }

   double Trans1D::max_wavenum() {
      return TransWrapper::max_wavenum(trans_dim);
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



} // end namespace




            


