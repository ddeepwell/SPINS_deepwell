/* Splits.hpp 

   Split an array over multple processors, along a single user-specified
   dimension.  Also includes a class to do the MPI-based transpose. */
#ifndef SPLITS_HPP
#define SPLITS_HPP 1

#include <vector>
#include <complex>
#include <map>
#include <blitz/array.h>
#include <mpi.h>

using namespace std;

vector<int> get_extents(int size, int numprocs);
vector<int> get_lbounds(int size, int numprocs);

using blitz::Array;
using blitz::TinyVector;
using blitz::GeneralArrayStorage;

/* Helper templates to find the right MPI-type automatically for common C++
   types */
template<typename t> class MPI_Typer {public: 
   static MPI_Datatype return_type();};
/* Default MPI datatypes for common base types */
#define MAKE_TYPE(t,tt) template<> class MPI_Typer<t> { \
   public: static MPI_Datatype return_type() { return tt; } };
MAKE_TYPE(int,MPI_INT)
MAKE_TYPE(float,MPI_FLOAT)
MAKE_TYPE(double,MPI_DOUBLE)
MAKE_TYPE(complex<double>,MPI_DOUBLE_COMPLEX)
MAKE_TYPE(complex<float>,MPI_COMPLEX)

class TransposeSpec {
   /* Data class with comparison, allowing us to "sort" the relevant aspects
      of a transpose.  Those are:
         Array sizes, in all three dimensions,
         Total number of processors involved in the transpose,
         Array dimension transposed from (existing split dimension), and
         Array dimension transposed to (desired split dimension) */
   public:
   int szx, szy, szz, numprocs, fromDim, toDim;
   TransposeSpec(int sx, int sy, int sz, int np, int fd, int td):
      szx(sx), szy(sy), szz(sz), numprocs(np), fromDim(fd), toDim(td) {
      }
   /* Helper preprocessor macro to make the less-than comparison a little
      faster to write */
#define COMPAREBY(x) if (x < rhs.x) return true; else if (x > rhs.x) return false;
   bool operator < (const TransposeSpec & rhs) {
      COMPAREBY(fromDim);
      COMPAREBY(toDim);
      COMPAREBY(numprocs);
      COMPAREBY(szx);
      COMPAREBY(szy);
      COMPAREBY(szz);
      return false;
   }
};

/* The bulk of the problem -- the class to actually do the transpose */
/* Ths class is templated both for generality and because there's bona-fide
   need for the generality.  In the fluids code, generally both Array<double>
   and Array<complex<double> > might need to be transposed, and they have
   different base types. */
template<class T> class Transposer {
   /* This object may very well contain temporary arrays, to speed up the
      transfer with MPI calls.  From experimentation, the transfer is by
      far the fastest if MPI can send contiguous chunks of data that is
      also in logical order.

      In English, that means a row-split array (split along the first dimension)
      is best sent as a Column Major Array (and received likewise).  Often,
      one of these assumptions will not hold.  Hence temporaries.

      Since temporaries are big, we don't want to make many of them.  Thus,
      we will use a static Map<TransposeSpec, Transposer<t> *> to cache
      these objects.  If we want to construct a new Transposer that matches
      one already constructed, recycle! */

   public:
   // Map disabled for now
//   static map<TransposeSpec, Transposer<t> *> * cache ;
//   int * refcount;
//   TransposeSpec us;
   
      int sizes[3]; // Full, non-split array sizes
      int num_proc, my_rank; // number of processors, my rank
      int from_split, to_split; // Dimensions that are orginially split, and
                                // split after the transpose
      int untouched_dim; // Dimension that is unchanged

      TinyVector<int,3> source_strides; // Optimal strides for source
      TinyVector<int,3> dest_strides; // Optimal strides for destination

      // Local lower bounds and extents for source/dest arrays
      TinyVector<int,3> source_lbound, source_extent;
      TinyVector<int,3> dest_lbound, dest_extent;

      // These should perhaps be ** for caching, later
      Array<T,3> * src_temp; // Temporary source and destination arrays
      Array<T,3> * dst_temp; // if necessary, for optimal transposes

      MPI_Datatype base_type; // Base type of transpose
      MPI_Datatype vec_type; // Type of base * length of untouched dimension
      GeneralArrayStorage<3> optim_order; // Optimal array storage order

      MPI_Comm comm; // MPI Communicator

      // Bases and extents for the split dimensions
      vector<int> base_from, base_to, extent_from, extent_to;

      // MPI derived datatypes for comunication
      MPI_Datatype send_type;
      vector<MPI_Datatype> rec_types;

      Transposer(int szx, int szy, int szz, int from_dim, int to_dim,
            MPI_Comm c = MPI_COMM_WORLD, 
            MPI_Datatype type = MPI_Typer<T>::return_type());

      // Perform the transpose
      void transpose(Array<T,3> & source, Array<T,3> & dest); 

      // Perform the reverse transpose -- source and dest have interchanged meanings
      void back_transpose(Array<T,3> & source, Array<T,3> & dest); 

      // Allocate temporary arrays if necessary
      bool issue_warning; // Issue performance warning if using temporaries
      void alloc_temps(Array<T,3> & source, Array<T,3> & dest,
            Array<T,3> * & src_pointer, Array<T,3> * & dst_pointer);

      // Helper functions to return the proper allocation parameters 
      // (with optimal storage order) when creating a split array
      void source_alloc(TinyVector<int,3> & lbound,
            TinyVector<int,3> & extent, GeneralArrayStorage<3> & order);
      void dest_alloc(TinyVector<int,3> & lbound,
            TinyVector<int,3> & extent, GeneralArrayStorage<3> & order);
      
};

#endif
