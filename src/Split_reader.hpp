#ifndef SPLIT_READER_HPP
#define SPLIT_READER_HPP 1
/* Reads part of a (2)D array on disk via mmap */

#include <blitz/array.h>

/* Reads a 2D array from a file via mmap and returns a copy of the requested
   slice in the freshly-allocated return array */
template <class T>
   blitz::Array<T,2> * read_2d_slice(const char * filename, // filename
                  int Nx, int Ny, // full sizes in x, y dimensions
                  blitz::Range range_x, // local requested range in x
                  blitz::Range range_y, // local requested range in y
                  blitz::GeneralArrayStorage<2> storage) ; // storage order

/* Specialization of above for no storage order */
template <class T>
   blitz::Array<T,2> * read_2d_slice(const char * filename,
                  int Nx, int Ny, blitz::Range range_x, blitz::Range range_y);
template <class T>
   blitz::Array<T,2> * read_2d_slice(const char * filename,
                  int Nx, int Ny, blitz::Range range_x, blitz::Range range_y) {
      blitz::GeneralArrayStorage<2> stor;
      return read_2d_slice<T>(filename,Nx,Ny,range_x,range_y,stor);

   }
/* Explicit template instantiation headers */
extern template blitz::Array<double,2> * read_2d_slice<double> (
      const char *, int, int, blitz::Range, blitz::Range,
      blitz::GeneralArrayStorage<2> );

extern template blitz::Array<float,2> * read_2d_slice<float> (
      const char *, int, int, blitz::Range, blitz::Range,
      blitz::GeneralArrayStorage<2> );

/* Include implementation */
#include "Split_reader_impl.cc"

#endif
