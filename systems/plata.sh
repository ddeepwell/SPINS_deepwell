#!/bin/bash

# System-specific settings for belize.math.uwaterloo.ca

CC=gcc
CXX=gicc
LD=mpic++

MPICXX=mpic++

# System-specific compiler flags
SYSTEM_CFLAGS="-m64"
SYSTEM_CXXFLAGS="-Wno-deprecated"
SYSTEM_LDFLAGS=-m64

# Compiler flags for debugging
DEBUG_CFLAGS="-g -DBZ_DEBUG"
DEBUG_LDFLAGS=

# Compiler flags for optimization
OPTIM_CFLAGS="-O3 -ffast-math -msse"
OPTIM_LDFLAGS=

# Compiler flags for extra optimization, such as -ip -ipo on icc
EXTRA_OPTIM_CFLAGS=
EXTRA_OPTIM_LDFLAGS=

# Library names/locations/flags for MPI-compilation.  This will
# probably not be necessary on systems with a working mpicc
# alias
MPI_CFLAGS=
MPI_LIB=
MPI_LIBDIR=
MPI_INCDIR=

# Library names/locations for LAPACK
LAPACK_LIB="-framework Accelerate"
LAPACK_LIBDIR=
LAPACK_INCDIR=

# Library locations for blitz; leave blank to use system-installed
# or compiled-with-this-package version
BLITZ_LIBDIR=
BLITZ_INCDIR=

# Library locations for fftw
FFTW_LIBDIR=
FFTW_INCDIR=

# Library locations for UMFPACK
UMF_INCDIR=
UMF_LIBDIR=

# Location/library for BLAS
BLAS_LIB="-framework Accelerate"
BLAS_LIBDIR=
BLAS_INCDIR=

