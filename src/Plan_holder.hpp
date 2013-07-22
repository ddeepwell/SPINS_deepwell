/* Plan_holder.hpp -- utility classes for building the cache of FFTW plans */
#ifndef PLAN_HOLDER_HPP // prevent double inclusion
#define PLAN_HOLDER_HPP 1

#ifdef __DEPRECATED
   #define _OLD_DEPRECATED
   #undef __DEPRECATED
#endif
#include <ext/hash_map>
#ifdef _OLD_DEPRECATED
   #define __DEPRECATED
#endif
#include <vector>
#include <assert.h>
#include <iostream>

namespace TArrayn {

using __gnu_cxx::hash_map;
using __gnu_cxx::hash;
using std::vector;
using std::cout; using std::endl;

class Plan_spec {
   /* FFTW requires transforms to be 'planned' in advance and executed later.
      This is how the library achieves most of its performance gains --
      otherwise expensive optimizations can be run once and then taken
      advantage of many times.  Unfortunately, it means that the transforms
      don't play nicely with an object-oriented perspective, where I'd like
      a transform to be an orthogonal operation on the array type.

      The quick-and-ugly solution is to create a plan each time we want to do
      a transform and delete the plan afterwards.  This is bad for obvious
      reasons.  Instead, we want to make the plan once and keep it in a cache;
      the plan itself requires negligible memory inside FFTW, so there aren't
      worries about total cache size.
      
      Step zero is to build a datatype that can uniquely identify a plan, given
      the constraints of SPINS operation. */

      public:
            const void * source; // Source and destination of transform
            const void * destination;
            const int type; // FFTW type flag
            const int dimension; // Dimension of transform

            Plan_spec(void * s, void * d, int t, int dim)
               :source(s), destination(d), type(t), dimension(dim) {}
            bool operator == (const Plan_spec & rhs) const {
               // Equality testing
               return (source == rhs.source && destination == rhs.destination
                     && type == rhs.type && dimension == rhs.dimension);
            }
};

struct plan_hash {
   /* An unfortunate ugliness of the STL extension hash_map is that the
      hash function must be overidden as a structure with operator(),
      rather than as a simple templated function. */
   size_t operator () (const Plan_spec &p) const {
      return hash<size_t>()((size_t) p.source + (size_t) p.destination) +
         hash<int>()(p.type * 10 + p.dimension);
   }
};


class Plan_holder {
   /* FFTW plans are, to quote the documentation, ``an opaque pointer type''.
   We need to keep them around for repeated transforms (the inner loops of
   SPINS), so we want to wrap the plans in a simple class.  The largest
   advantage is that user code will not have to worry at all about properly
   destroying the plans, since it's simple to wrap them up in a reference-
   counted container (this class).  Also, we can entirely wrap the
   fftw_execute_plan function in an object-oriented manner.

   One niggling detail is that transforms of 3D arrays aren't always simple.
   If the dimension to be transformed has the smallest stride, it's easy --
   the transform becomes that of many 1D arrays, interleaved one right after
   the prior.  If the stride is the largest, then the case is also simple --
   there are many arrays, but with a single large stride between elements and
   a single small distance (probably 1) between each array.

   The intermediate case is more difficult, however -- a single stride/distance
   pair is not sufficient.  Thus, we must store and execute many plans, each
   transforming (properly) a 2D slice of the 3D array.  We can wrap that in this
   class by using a vector, allowing user code to only care about ->execute().
   */
   private:
      vector<fftw_plan> plans;
      int * refcount;
      void _check_and_delete() {
         if (refcount && *refcount == 0) { // Last reference to the plans
            delete refcount; refcount = 0;
            for(vector<fftw_plan>::iterator i = plans.begin();
                  i < plans.end(); i++) {
               fftw_destroy_plan(*i);
            }
         }
      }
   public:
      Plan_holder() // Default constructor -- empty plans
         :plans(0), refcount(0) {}
      Plan_holder(fftw_plan singleplan) // Single plan constructor
         :plans(1) {
            plans[0] = singleplan;
            refcount = new int(1);
         }
      Plan_holder(vector<fftw_plan> manyplans)
         :plans(manyplans) { // copy plans 
            refcount = new int(1);
         }
      Plan_holder(const Plan_holder & copyfrom) { // Copy constructor
         refcount = copyfrom.refcount;
         if (refcount) (*refcount)++;
         plans = copyfrom.plans;
      }
      ~Plan_holder() { // destructor
         if (refcount) (*refcount)--;
         _check_and_delete();
      }
      Plan_holder & operator = (const Plan_holder & rhs) { // assignment
         if (refcount) {
            (*refcount)--; 
            _check_and_delete();
         }
         refcount = rhs.refcount;
         if (refcount) (*refcount)++;
         plans = rhs.plans;
         return (*this);
      }
      void execute() { // Execute all contained plans
         for(vector<fftw_plan>::iterator i = plans.begin();
               i < plans.end(); i++) {
            fftw_execute(*i);
         }
      }
};

} // end namespace
#endif
