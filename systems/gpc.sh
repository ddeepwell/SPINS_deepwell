#!/bin/bash

# System-specific settings for gpc, a scinet cluster

CC=icc
CXX=icpc
LD=mpicxx

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
MPICXX=mpicxx
MPI_CFLAGS=
MPI_LIB=
MPI_LIBDIR=
MPI_INCDIR=

# Library names/locations for LAPACK
LAPACK_LIB="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core"
LAPACK_LIBDIR=-L"$MKLPATH"
LAPACK_INCDIR=

# Library locations for blitz; leave blank to use system-installed
# or compiled-with-this-package version
BLITZ_LIBDIR=
BLITZ_INCDIR=

# Library locations for fftw
#FFTW_LIBDIR=-L/uv-global/software/fftw/fftw-3.3.1/lib
#FFTW_INCDIR=-I/uv-global/software/fftw/fftw-3.3.1/include

# Library locations for UMFPACK
UMF_INCDIR=
UMF_LIBDIR=

# Location/library for BLAS
# The sharcnet clusters are strange, and their compiler
# script includes the blas libraries with -llapack
BLAS_LIB=$LAPACK_LIB
BLAS_LIBDIR=$LAPACK_LIBDIR
BLAS_INCDIR=

