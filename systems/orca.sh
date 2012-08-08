#!/bin/bash

# System-specific settings for the orca.sharcnet.ca cluster

CC=mpicc
CXX=mpic++
LD=mpic++

# System-specific compiler flags
SYSTEM_CFLAGS="-Wall -wd981 -wd444 -wd1572 -wd383 -wd869"
SYSTEM_LDFLAGS=

# Compiler flags for debugging
DEBUG_CFLAGS="-g -DBZ_DEBUG"
DEBUG_LDFLAGS=

# Compiler flags for optimization
OPTIM_CFLAGS="-O3 -fp-model fast=2"
OPTIM_LDFLAGS=

# Compiler flags for extra optimization, such as -ip -ipo on icc
EXTRA_OPTIM_CFLAGS="-ip -ipo"
EXTRA_OPTIM_LDFLAGS=

# Library names/locations/flags for MPI-compilation.  This will
# probably not be necessary on systems with a working mpicc
# alias
MPI_CFLAGS=
MPI_LIB=-lmpi
MPI_LIBDIR=
MPI_INCDIR=

# Library names/locations for LAPACK
LAPACK_LIB="-llapack"
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
# The sharcnet clusters are strange, and their compiler
# script includes the blas libraries with -llapack
BLAS_LIB="-llapack"
BLAS_LIBDIR=
BLAS_INCDIR=

