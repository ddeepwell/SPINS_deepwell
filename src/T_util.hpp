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
    void write_array_old(blitz::Array<double, 3>  const & ar, const std::string basename,
            int seq_num = -1, MPI_Comm c = MPI_COMM_WORLD);
    /* Create .m matlab file to read a written array in as a proper MATLAB array
       with the same semanticcs */
    void write_reader(blitz::Array<double, 3> const & ar, const std::string basename, 
            bool seq = false, MPI_Comm c = MPI_COMM_WORLD);

    /* Read from an array written via write_array to an appropriate (subset)
       processor-local array.  Required for restarting. */
    void read_array_old(blitz::Array<double,3> & ar, const char * filename,
            int size_x, int size_y, int size_z, MPI_Comm c = MPI_COMM_WORLD);

    void convert_index_2(int n, int Nx, int Ny, int Nz, int * I, int * J, int * K);

    // Parallel version of read_array
    void read_array(TArrayn::DTArray & ar, const char * filename,
            int size_x, int size_y, int size_z, MPI_Comm c = MPI_COMM_WORLD);

    // Parallel version of write_array
    void write_array(TArrayn::DTArray & ar, char * basename,
            int seq_num = -1, MPI_Comm c = MPI_COMM_WORLD);

    // Initialize temporary files for chains
    void initialize_chain_tmps(char* varname, MPI_File ** files,
            double *** chain_coords, int * num_chains, MPI_Comm = MPI_COMM_WORLD);

    // Initialize files for chains
    void initialize_chain_finals(char* varname, MPI_File ** files,
            double *** chain_coords, int * num_chains, MPI_Comm = MPI_COMM_WORLD);

    // Write data to chain temporary files
    void write_chains_2(DTArray & var, MPI_File ** files,
            double *** chain_coords, int * num_chains,
            int Nx, int Ny, int Nz, int chain_count,
            double * x_chain_data_buffer, double * y_chain_data_buffer,
            double * z_chain_data_buffer, MPI_Comm c = MPI_COMM_WORLD);

    // Stitch temporary files into final files
    void stitch_chains_2(char* varname, MPI_File ** files_from, MPI_File ** files_to, int * num_chains,
            int Nx, int Ny, int Nz, double *** chain_coords, int chain_write_count,
            int prev_chain_write, int lb, int ub, MPI_Comm c = MPI_COMM_WORLD);

    // Initialize temporary files for slicess
    void initialize_slice_tmps(char* varname, MPI_File ** files,
            double ** slice_coords, int * num_slices, MPI_Comm c = MPI_COMM_WORLD);

    // Initialize files for slices
    void initialize_slice_finals(char* varname, MPI_File ** files,
            double ** slice_coords, int * num_slices, MPI_Comm c = MPI_COMM_WORLD);

    // Write data to slice temporary files
    void write_slices_2(DTArray & var, MPI_File ** files,
            double ** slice_coords, int * num_slices,
            int Nx, int Ny, int Nz, int slice_count, MPI_Comm c = MPI_COMM_WORLD);

    // Stitch temporary files into final files
    void stitch_slices_2(char* varname, MPI_File ** files_from, MPI_File ** files_to, int * num_slices,
            int Nx, int Ny, int Nz, double ** slice_coords, int slice_write_count,
            int prev_slice_write, int lb, int ub, MPI_Comm c = MPI_COMM_WORLD);



} // end namespace
#endif
