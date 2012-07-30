#ifndef PAR_UTIL_HPP
#define PAR_UTIL_HPP 1

#include "TArray.hpp"
#include <mpi.h>
#include <blitz/array.h>
#include "Splits.hpp"

::TArrayn::DTArray * alloc_array(int szx, int szy, int szz, 
                           MPI_Comm comm = MPI_COMM_WORLD);

/* For other allocation of split arrays, the various component
   pieces are wrapped here */
blitz::TinyVector<int,3> alloc_lbound(int szx, int szy, int szz,
                           MPI_Comm comm = MPI_COMM_WORLD);
blitz::TinyVector<int,3> alloc_extent(int szx, int szy, int szz,
                           MPI_Comm comm = MPI_COMM_WORLD);
blitz::GeneralArrayStorage<3> alloc_storage(int szx, int szy, int szz,
                           MPI_Comm comm = MPI_COMM_WORLD);

blitz::Range split_range(int sz, MPI_Comm comm = MPI_COMM_WORLD);

// Quick check of whether this is the master process
bool master(MPI_Comm comm = MPI_COMM_WORLD);


/* Helper functions to simplify the MPI_Reduce(sum/max/all) problem */

// Scalar functions
template <class T> T psmax(T x, MPI_Comm c = MPI_COMM_WORLD,
      MPI_Datatype type = MPI_Typer<T>::return_type()) ;
template <class T> T psmin(T x, MPI_Comm c = MPI_COMM_WORLD,
      MPI_Datatype type = MPI_Typer<T>::return_type()) ;
template <class T> T pssum(T x, MPI_Comm c = MPI_COMM_WORLD,
      MPI_Datatype type = MPI_Typer<T>::return_type()) ;
template <class T> bool psany(T x, MPI_Comm c = MPI_COMM_WORLD) ;
template <class T> bool psall(T x, MPI_Comm c = MPI_COMM_WORLD) ;

/* Vector operations */
template <class T, int d> T pvmax(const blitz::Array<T,d> & x, 
      MPI_Comm c = MPI_COMM_WORLD,
      MPI_Datatype type = MPI_Typer<T>::return_type()) ;
template <class T, int d> T pvmin(const blitz::Array<T,d> & x, 
      MPI_Comm c = MPI_COMM_WORLD,
      MPI_Datatype type = MPI_Typer<T>::return_type()) ;
template <class T, int d> T pvsum(const blitz::Array<T,d> & x, 
      MPI_Comm c = MPI_COMM_WORLD,
      MPI_Datatype type = MPI_Typer<T>::return_type()) ;
/* Prototype for explicitly-instantiated templates? */
extern template double pvmax(const blitz::Array<double,3> &, MPI_Comm, MPI_Datatype);
extern template double pvsum(const blitz::Array<double,3> &, MPI_Comm, MPI_Datatype);
extern template int pvmax(const blitz::Array<int,3> &, MPI_Comm, MPI_Datatype);
extern template int pvsum(const blitz::Array<int,3> &, MPI_Comm, MPI_Datatype);

#include "Par_util_impl.cc"
#endif
