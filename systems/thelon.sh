#!/bin/bash

# System specific flags for thelon.math.uwaterloo.ca

###################
### PLEASE NOTE ###
###################

# As of this writing (28 Nov 2013), consider this system-
# specific configuration file for thelon to be PROVISIONAL
# at best.  The system is missing a few key libraries that
# make a general install difficult, however they can be
# installed individually.

# *) There is no system-wide MPI library, so OpenMPI needs
#    to be installed in the home directory; this also 
#    supplies the mpirun and mpicc/mpic++ wrappers.

# *) The system-wide LAPACK doesn't work tremendously well
#    after the system upgrade to gcc4.4.7; ATLAS can be
#    installed in the home directory (with a standard
#    netlib tarball supplied for the missing functions)
#    to fill the gap.

# In both cases, the library settings listed here match
# the situation in ~csubich/{bin,include,lib}

CC=gcc
CXX=g++
LD=mpic++

# System-specific compiler flags
SYSTEM_CFLAGS=
SYSTEM_LDFLAGS=
SYSTEM_CXXFLAGS=

# Compiler flags for debugging
DEBUG_CFLAGS="-g -DBZ_DEBUG"
DEBUG_LDFLAGS=

# Compiler flags for optimization
OPTIM_CFLAGS="-O3 -ffast-math"
OPTIM_LDFLAGS=$OPTIM_CFLAGS

# Compiler flags for extra optimization, such as -ip -ipo on icc
EXTRA_OPTIM_CFLAGS=
EXTRA_OPTIM_LDFLAGS=

# Library names/locations/flags for MPI-compilation.  This will
# probably not be necessary on systems with a working mpicc
# alias
MPICXX=mpic++
MPI_CFLAGS=
MPI_LIB=
MPI_LIBDIR=
MPI_INCDIR=

# Library names/locations for LAPACK
LAPACK_LIB="-llapack -lf77blas -lcblas -latlas -lgfortran"
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
BLAS_LIB="-lf77blas -lcblas -latlas -lgfortran"
BLAS_LIBDIR=
BLAS_INCDIR=

