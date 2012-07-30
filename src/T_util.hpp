#ifndef T_UTIL_HPP
#define T_UTIL_HPP 1

#include "TArray.hpp"
#include "Parformer.hpp"
#include <mpi.h>
#include <blitz/array.h>
#include <string>

namespace TArrayn {

   using namespace Transformer;
   
/* Real-to-complex Fourier derivative */
void deriv_fft(DTArray & source, Trans1D & tform, blitz::Array<double, 3> & dest);

/* Cosine derivative (DCT10) */
void deriv_dct(DTArray & source, Trans1D & tform, blitz::Array<double, 3> & dest);

/* Sine derivative (DST10), for symmetry with cosine */
void deriv_dst(DTArray & source, Trans1D & tform, blitz::Array<double, 3> & dest);

/* Chebyshev derivative */ 
void deriv_cheb(DTArray & source, Trans1D & tform, blitz::Array<double, 3> & dest);

/* Spectral filtering, with sensible defaults */
void filter3(DTArray & source, TransWrapper & tform, 
      S_EXP dim1_type, S_EXP dim2_type, S_EXP dim3_type, 
      double cutoff=0.7, double order = 4, double strength = 20);

/* Array-to-file writer, for MATLAB input */
void write_array(blitz::Array<double, 3>  const & ar, const std::string basename, 
      int seq_num = -1, MPI_Comm c = MPI_COMM_WORLD);
/* Create .m matlab file to read a written array in as a proper MATLAB array
   with the same semanticcs */
void write_reader(blitz::Array<double, 3> const & ar, const std::string basename, 
      bool seq = false, MPI_Comm c = MPI_COMM_WORLD);

/* Read from an array written via write_array to an appropriate (subset)
   processor-local array.  Required for restarting. */
void read_array(blitz::Array<double,3> & ar, const char * filename,
      int size_x, int size_y, int size_z, MPI_Comm c = MPI_COMM_WORLD);

} // end namespace
#endif
