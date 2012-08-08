#!/bin/bash

# make_deps.sh : builds dependency libraries for SPINS

# The SPINS model requires several external libraries for compilation,
# some of which are non-standard.  They are:

# Blitz++ -- blitz.sourceforge.net
# This is a C++ "meta-template library" for arrays that allows
# manipulation of multidimensional arrays in C++ code with a
# MATLAB-like syntax
BUILD_BLITZ=no 

# fftw -- www.fftw.org
# This is a self-contained library for high-performance Fast
# Fourier Transforms
BUILD_FFTW=no

# UMFPACK -- www.cise.ufl.edu/research/sparse/umfpack
# AMD     -- www.cise.ufl.edu/research/sparse/amd/
# UFconfig-- www.cise.ufl.edu/research/sparse/SuiteSparse_config
#
# These are libraries required for the sparse direct solve used
# at the corsest level of the 2D geometric multigrid algorithm
# in SPINS

BUILD_UMFPACK=yes

# Read in the appropriate system script.  If none is specified on the
# command line, guess based on the hostname

if [ $# -gt 0 ]; then
   if [ -f $1 ]; then
      echo Reading system-specific variables from $1
      source $1
   elif [ -f systems/$1.sh ]; then
      echo Reading system-specific variables from systems/$1.sh
      source systems/$1.sh
   else 
      echo Neither $1 nor /systems/$1.sh are found!
      exit 1
   fi
else
   echo Guessing that system-specific variables are in systems/`hostname -s`.sh
   source systems/`hostname -s`.sh || (echo "... but they're not"; exit 1)
fi

# Current working directory
CWD=`pwd`


# Make, if necessary, the local include and lib directories
if [ ! -d lib ]; then mkdir lib; fi
if [ ! -d include ]; then mkdir include; fi

if [ ! "$BUILD_BLITZ" = "yes" ]; then 
	echo "NOT building Blitz++"
else
	echo "Building Blitz++"
	# Download the Blitz tarball if necessary
	if [ ! -e "blitz_2010.tgz" ]; then
		wget http://belize.math.uwaterloo.ca/~csubich/redist/blitz_2010.tgz
	fi
	(tar -xzvf blitz_2010.tgz > /dev/null) || (echo "Untar of Blitz FAILED"; exit 1);
	pushd blitz
	(./configure --prefix="$CWD" --disable-fortran > /dev/null) && \
		(make lib > /dev/null) && \
		pushd blitz && (make install > /dev/null) && popd  && \
		pushd lib && (make install > /dev/null) && popd  && \
		pushd random && (make install > /dev/null) && popd && \
	popd || (echo "Could not compile/install Blitz"; exit 1)
	echo "Blitz++ built!"
fi

if [ ! "$BUILD_FFTW" = "yes" ]; then
	echo "NOT building FFTW"
else
	echo "Building FFTW"
	# Download FFTW if necessary
	if [ ! -e "fftw-3.3.2.tar.gz" ]; then
		wget http://belize.math.uwaterloo.ca/~csubich/redist/fftw-3.3.2.tar.gz
	fi
	(tar -xzvf fftw-3.3.2.tar.gz > /dev/null) || (echo "Untar of FFTW FAILED"; exit 1)
	pushd fftw-3.3.2
	(./configure --prefix="$CWD" --disable-fortran --enable-sse2 > /dev/null) && \
		(make > /dev/null) && \
		(make install-libLTLIBRARIES > /dev/null) && \
		pushd api; (make install > /dev/null) && popd \
		|| (echo "Could not compile/install FFTW!"; exit 1);
	popd
	echo "FFTW built!"
fi


if [ ! "$BUILD_UMFPACK" = "yes" ]; then
	echo "NOT building UMFPACK"
else
	echo "Building UMFPACK"
	# Download UFconfig
	if [ ! -e "UFconfig-3.4.0.tar.gz" ]; then
		wget http://belize.math.uwaterloo.ca/~csubich/redist/UFconfig-3.4.0.tar.gz
	fi
	if [ ! -e "UMFPACK.tar.gz" ]; then
		wget http://belize.math.uwaterloo.ca/~csubich/redist/UMFPACK.tar.gz
	fi
	if [ ! -e "AMD.tar.gz" ]; then
		wget http://belize.math.uwaterloo.ca/~csubich/redist/AMD.tar.gz
	fi

	# Untar the lot
	(tar -xzvf UFconfig-3.4.0.tar.gz;
	 tar -xzvf UMFPACK.tar.gz;
	 tar -xzvf AMD.tar.gz;) > /dev/null || (echo "Could not untar UMFACK"; exit 1)
	
	# There is no nice ./configure script, so we have to "edit" the UFconfig
	# makefile by hand (that controls the others).

	pushd UFconfig
	cat UFconfig.mk | sed \
	   -e "s/^CC.*/CC = ${CC}/" \
 	   -e "s/^CPLUSPLUS.*/CPLUSPLUS = ${CXX}/" \
	   -e "s/^BLAS.*/BLAS = ${BLAS_INCDIR} ${BLAS_LIBDIR} ${BLAS_LIB}/" \
	   -e "s/^LAPACK.*/LAPACK = ${LAPACK_INCDIR} ${LAPACK_LIBDIR} ${LAPACK_LIB}/" \
	   > UFconfig.new
	mv UFconfig.new UFconfig.mk
	popd

	echo "Building AMD"
	pushd AMD
	(make > /dev/null) || (echo "Could not make AMD"; exit 1)
	cp Include/* ../include/
	cp Lib/* ../lib
	popd

	pushd UMFPACK
	echo "Building UMFPACK"
	(make library > /dev/null) || (echo "Could not make UMFPACK"; exit 1);
	cp Include/* ../include
	cp Lib/* ../lib
	popd

   cp UFconfig/UFconfig.h ./include/
	echo "Done!"
fi

