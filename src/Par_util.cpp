#include "Par_util.hpp"
#include "Splits.hpp"
#include <blitz/array.h>
#include <vector>

using namespace std;
using namespace blitz;
using TArrayn::DTArray;

/* Collection of utilities for parallel code */

DTArray * alloc_array(int szx, int szy, int szz, MPI_Comm comm) {
   /* Allocates and returns a DTArray, properly split and StorageOrdered,
      for a given array size */
   
   GeneralArrayStorage<3> order;
   int myrank, numproc;
   MPI_Comm_size(comm,&numproc);
   MPI_Comm_rank(comm,&myrank);

   if (numproc > 1) { // Single-processor runs use default C-order
      order.ordering()[0] = secondDim;
      order.ordering()[1] = firstDim;
      order.ordering()[2] = thirdDim;
   }

   vector<int> xlb = get_lbounds(szx,numproc);
   vector<int> xex = get_extents(szx,numproc);

   TinyVector<int,3> lbounds, extents;
   lbounds(0) = xlb[myrank]; lbounds(1) = 0; lbounds(2) = 0;
   extents(0) = xex[myrank]; extents(1) = szy; extents(2) = szz;
   return new DTArray(lbounds,extents,order);
}

GeneralArrayStorage<3> alloc_storage(int szx, int szy, int szz, MPI_Comm c) {
   GeneralArrayStorage<3> order;
   int myrank, numproc;
   MPI_Comm_size(c,&numproc);
   MPI_Comm_rank(c,&myrank);

   if (numproc > 1) { // Single-processor runs use default C-order
      order.ordering()[0] = secondDim;
      order.ordering()[1] = firstDim;
      order.ordering()[2] = thirdDim;
   }
   return order;
}

TinyVector<int,3> alloc_lbound(int szx, int szy, int szz, MPI_Comm c) {
   int myrank, numproc;
   MPI_Comm_size(c,&numproc);
   MPI_Comm_rank(c,&myrank);
   vector<int> xlb = get_lbounds(szx,numproc);

   TinyVector<int,3> lbounds;
   lbounds(0) = xlb[myrank]; lbounds(1) = 0; lbounds(2) = 0;
   return lbounds;
}

TinyVector<int,3> alloc_extent(int szx, int szy, int szz, MPI_Comm c) {
   int myrank, numproc;
   MPI_Comm_size(c,&numproc);
   MPI_Comm_rank(c,&myrank);
   vector<int> xex = get_extents(szx,numproc);

   TinyVector<int,3> extents;
   extents(0) = xex[myrank]; extents(1) = szy; extents(2) = szz;
   return extents; 
}

bool master(MPI_Comm c) {
   static MPI_Comm last_comm = MPI_COMM_WORLD;
   static int myrank = -1;
   if (c != last_comm || myrank < 0) {
      last_comm = c;
      MPI_Comm_rank(last_comm,&myrank);
   }
   return (myrank == 0);
}


blitz::Range split_range(int sz, MPI_Comm comm) {
   /* Returns a Range(lb,ub) for a dimension split over multiple processors*/
   int myrank, numproc;
   MPI_Comm_size(comm,&numproc);
   MPI_Comm_rank(comm,&myrank);

   vector<int> xlb = get_lbounds(sz,numproc);
   vector<int> xex = get_extents(sz,numproc);

   return blitz::Range(xlb[myrank],xlb[myrank]+xex[myrank]-1);
}
/* Explicit instantiation of reductions */
template double psmax<double>(double, MPI_Comm, MPI_Datatype);
template double psmin<double>(double, MPI_Comm, MPI_Datatype);
template double pssum<double>(double, MPI_Comm, MPI_Datatype);
template int psmax<int>(int, MPI_Comm, MPI_Datatype);
template int psmin<int>(int, MPI_Comm, MPI_Datatype);
template int pssum<int>(int, MPI_Comm, MPI_Datatype);


template double pvmax<double,3>(const blitz::Array<double,3> &, 
      MPI_Comm, MPI_Datatype);
template double pvsum<double,3>(const blitz::Array<double,3> &, 
      MPI_Comm, MPI_Datatype);
template int pvmax<int,3>(const blitz::Array<int,3> &, 
      MPI_Comm, MPI_Datatype);
template int pvsum<int,3>(const blitz::Array<int,3> &, 
      MPI_Comm, MPI_Datatype);
