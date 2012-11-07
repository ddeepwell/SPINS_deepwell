#!/bin/bash

# System-specific settings for the orca.sharcnet.ca cluster

CC=icc
CXX=icpc
LD=mpic++

# System-specific compiler flags
SYSTEM_CFLAGS=

SYSTEM_LDFLAGS=

# Compiler flags for debugging
DEBUG_CFLAGS="-g -DBZ_DEBUG"
DEBUG_LDFLAGS=

# Compiler flags for optimization
OPTIM_CFLAGS="-O3 -fp-model fast=2"
OPTIM_LDFLAGS=$OPTIM_CFLAGS

# Compiler flags for extra optimization, such as -ip -ipo on icc
EXTRA_OPTIM_CFLAGS="-ip -ipo"
EXTRA_OPTIM_LDFLAGS=$EXTRA_OPTIM_CFLAGS

# Library names/locations/flags for MPI-compilation.  This will
# probably not be necessary on systems with a working mpicc
# alias
MPICXX=mpic++
MPI_CFLAGS=
MPI_LIB=
MPI_LIBDIR=
MPI_INCDIR=

# Library names/locations for LAPACK
LAPACK_LIB="-llapack"
LAPACK_LIBDIR=-L`dirname $(find /opt/sharcnet/intel/11* -name 'libmkl_core.a' | head -1)`
LAPACK_INCDIR=

# Library locations for blitz; leave blank to use system-installed
# or compiled-with-this-package version
BLITZ_LIBDIR=
BLITZ_INCDIR=

# Library locations for fftw
FFTW_LIBDIR=-L`dirname $(find /opt/sharcnet/fftw* -name 'libfftw3.a' | grep intel | head -1)`
FFTW_INCDIR=-I`dirname $(find /opt/sharcnet/fftw* -name "fftw3.h" | grep intel | head -1)`

# Library locations for UMFPACK
UMF_INCDIR=
UMF_LIBDIR=

# Location/library for BLAS
# The sharcnet clusters are strange, and their compiler
# script includes the blas libraries with -llapack
BLAS_LIB="-llapack"
BLAS_LIBDIR=
BLAS_INCDIR=

