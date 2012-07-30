#include "Splits.hpp"

#include <vector>
#include <blitz/array.h> 
#include <blitz/tinyvec-et.h>
#include <stdio.h>
#include <iostream>
#include <complex>

using namespace std;
using namespace blitz;


vector<int> get_extents(int size, int numprocs) {
   /* Gets the extent size per processor (0-numprocs-1) when an array dimension
      of length size is split */
   vector<int> extents(numprocs);
   /* The split is as close to even as possible, with the first few processors
      dividing any excess. */
   for (int i = 0; i < numprocs; i++) {
      extents[i] = size/numprocs + (i < (size % numprocs) );
   }
   return extents;
}

vector<int> get_lbounds(int size, int numprocs) {
   /* Gets the lower bound per processor of an array of length size is split
      over numprocs processors.  The cumulative sum (starting with 0) of
      extents. */
   vector<int> lbounds(numprocs);
   for (int i = 0; i < numprocs; i++) {
      lbounds[i] = i*(size/numprocs) + min(i,size % numprocs);
   }
   return lbounds;
}

template <class T>
Transposer<T>::Transposer(int szx, int szy, int szz, int from_dim, int to_dim,
      MPI_Comm c, MPI_Datatype t):
src_temp(0), dst_temp(0), issue_warning(true) {
   comm = c; base_type = t;
   /* Find out about the parallel environment */
   MPI_Comm_size(comm,&num_proc);
   MPI_Comm_rank(comm,&my_rank);


   // Store sizes
   sizes[0] = szx; sizes[1] = szy; sizes[2] = szz;

   /* Assign dimensions to roles */
   from_split = from_dim;
   to_split = to_dim;
   untouched_dim = firstDim + secondDim + thirdDim - from_split - to_split;
   if (num_proc == 1) {
      /* If we're running only a single processor, use default C-storage order */
      optim_order = GeneralArrayStorage<3>(); // default
   } else {
      optim_order.ordering()[0] = untouched_dim;
      optim_order.ordering()[1] = from_dim;
      optim_order.ordering()[2] = to_dim;
   }

   extent_from = get_extents(sizes[from_dim],num_proc);
   extent_to = get_extents(sizes[to_dim], num_proc);
   base_from = get_lbounds(sizes[from_dim], num_proc);
   base_to = get_lbounds(sizes[to_dim], num_proc);

   /* Compute local lower bounds and extents for arrays */
   source_lbound(from_dim) = base_from[my_rank];
   source_lbound(to_dim) = 0;
   source_lbound(untouched_dim) = 0;
   source_extent(from_dim) = extent_from[my_rank];
   source_extent(to_dim) = sizes[to_dim];
   source_extent(untouched_dim) = sizes[untouched_dim];
   
   dest_lbound(to_dim) = base_to[my_rank];
   dest_lbound(from_dim) = 0;
   dest_lbound(untouched_dim) = 0;
   dest_extent(to_dim) = extent_to[my_rank];
   dest_extent(from_dim) = sizes[from_dim];
   dest_extent(untouched_dim) = sizes[untouched_dim];

/*   cout << "Local source array -- " << source_lbound << " x " <<
      source_extent << endl;
   cout << "Local dest array -- " << dest_lbound << " x " <<
      dest_extent << endl;*/
   
   /* Copute strides */
   /* In order to send contigious data (which seems fastest for MPI),
      the major dimension is the one that is to be split, and the
      "most minor" dimension is the one untouched. */
   source_strides(untouched_dim) = 1;
   dest_strides(untouched_dim) = 1;
   source_strides(from_dim) = dest_strides(from_dim) = sizes[untouched_dim];
   source_strides(to_dim) = sizes[untouched_dim]*extent_from[my_rank];
   dest_strides(to_dim) = sizes[untouched_dim]*sizes[from_dim];

   /* Create derived MPI Datatypes for the transpose */

   // Create vector datatype
   MPI_Type_contiguous(sizes[untouched_dim],base_type,&vec_type);
   MPI_Type_commit(&vec_type);

   // Now, we're sending contiguous data.  With the prototype in mind of
   // transposing an x-split array into a y-split array (row-split matrix
   // into a column split matrix), each process will be sending some
   // (recipient-dependent, generally) number of partial rows to each
   // other processor.  Thus, we need the "partial row" sendtype

   MPI_Type_contiguous(extent_from[my_rank],vec_type,&send_type);
   MPI_Type_commit(&send_type);

   // For the receiving side, we have to use multiple datatypes in the
   // general case.  We'll receive a -fixed number- of partial rows from
   // each other process, but the size of that partial row will itself vary
   // by process.  This calls for a relatively complicated MPI Datatype,
   // encompassing the entire partial row plus a false "upper bound" so that
   // there's a gap.

   rec_types.resize(num_proc);  // make sure our vector can hold everything

   for (int i = 0; i < num_proc; i++) {
      // We'll have to use the MPI_Type_struct syntax, so build arrays:
      MPI_Datatype types[2] = {vec_type,MPI_UB};
      int counts[2]; counts[0] = extent_from[i]; counts[1] = 1;
      // Now, set displacements to set the "end" of the datatype a full
      // "row" ahead of the partial row received.
      MPI_Aint displs[2] = {0,sizeof(T)*sizes[untouched_dim]*sizes[from_dim]};

      // And make the type
      MPI_Type_struct(2,counts,displs,types,&rec_types[i]);
      MPI_Type_commit(&rec_types[i]);
      /* Impelementation note: one optimization possible here is to collapse
         this typelist.  This impelementation makes no assumptions about how
         many part-rows are received per processor, but our default splitting
         arrangement (above) splits things nearly evenly.  If we assume that
         splitting, then we only need a maximum of two types.  C'est la vie.*/
   }
}
template <class T>
void Transposer<T>::alloc_temps(Array<T,3> & source, Array<T,3> & dest,
      Array<T,3> * & src_pointer, Array<T,3> * & dst_pointer) {
   // Check to see if the source and destination arrays are in an optimal
   // ordering.  If so, then fine.  If not, copy them over to (perhaps
   // freshly-allocated) temporary arrays, so that the MPI goes faster.
   if (!all(source_strides == source.stride())) {
      /* We don't have an optimal order, so copy to the temporary */
      if (!src_temp) {
         src_temp = new Array<T,3> (source_lbound,source_extent,optim_order);
      }
      *src_temp = source;
      src_pointer = src_temp;
      if (issue_warning) {
         if (!my_rank)
            cerr << "Warning: using temporary arrays for transpose.  This is probably very" <<
             "bad for performance\n";
         issue_warning = false;
      }
   } else {
      src_pointer = &source;
   }
   if (!all(dest_strides == dest.stride())) {
      /* We don't have an optimal order, so copy to the temporary */
      if (!dst_temp) {
         dst_temp = new Array<T,3> (dest_lbound,dest_extent,optim_order);
      }
      dst_pointer = dst_temp;
      if (issue_warning) {
         cerr << "Warning: using temporary arrays for transpose.  This is probably very" <<
            "bad for performance\n";
         issue_warning = false;
      }
   } else {
      dst_pointer = &dest;
   }
}
   
template <class T>
void Transposer<T>::transpose(Array<T,3> & source, Array<T,3> & dest) {
   /* Actually perform the transpose, using MPI calls */

   // Pointers to indirect the source and dest arrays, if we need to use
   // temporaries
   /* First, if we only have one processor, just copy */
   if (num_proc == 1) {
      dest = source;
      return;
   }
   Array<T,3> * source_pointer, * dest_pointer; 

   assert(all(source.extent() == source_extent) &&
         all(source.lbound() == source_lbound));
   assert(all(dest.extent() == dest_extent) &&
         all(dest.lbound() == dest_lbound));

   // See temporary-allocation for the logic of this
   alloc_temps(source, dest, source_pointer, dest_pointer);

   // Create references, so that the transpoce can use the more familiar
   // array(i,j,k) notation without tons of pointer dereferences cluttering
   // things up.
   Array<T,3> & srcarray = *source_pointer;
   Array<T,3> & dstarray = *dest_pointer;

   // Finally, where's the beef!?
   TinyVector<int,3> send_location, rec_location;
   send_location(untouched_dim) = rec_location(untouched_dim) = 0;
   send_location(from_split) = source_lbound(from_split);
   rec_location(to_split) = dest_lbound(to_split);

   for (int k = 0; k < num_proc; k++ ) {
      /* Since MPI_Alltoallw is not well-supported, we fake it with sendrecv
         calls.  Starting with an offset of 0, each process will send to
         the one (offset) above it [and thus receive from the one (offset)
         below it], wrapping around when we reach 0 / numproc. */
      int recfrom = (num_proc + my_rank - k) % num_proc;
      int sendto = (num_proc + my_rank + k) % num_proc;

      // Now, pick out the proper locations to send from and receive to.
      // We send from the "columns" that the destination processor needs:
      send_location(to_split) = base_to[sendto];
      int send_count = extent_to[sendto];
      // And likewise for what we receive:
      rec_location(from_split) = base_from[recfrom];
      int rec_count = dest_extent(to_split);

      MPI_Status ignoreme;

      // And make the sendrecv call:
     MPI_Sendrecv(&srcarray(send_location),send_count,send_type,sendto,0,
            &dstarray(rec_location),rec_count,rec_types[recfrom],recfrom,0,
            comm,&ignoreme);
      
//            printf( "Sending %d partial rows to %d, receiving %d partial rows from %d\n",send_count,sendto,rec_count,recfrom);
   }
   // If we transposed into the temporary, copy back
   if (&dstarray != &dest) {
      dest = dstarray;
   }
}
template <class T>
void Transposer<T>::back_transpose(Array<T,3> & r_source, Array<T,3> & r_dest) {
   /* Actually perform the reverse transpose, using MPI calls */

   /* Conceptually, we've built the class with a preferred direction, "source" to "destination",
      but the relevant MPI calls are completely reversible.  That is, with the datatypes we're
      using, it should be just as efficient to go backwards as forwards.  Hence,
      rever_transpose, with r_source (reverse source) and r_dest (reverse destination) */

   /* First, if we only have one processor, just copy */
   if (num_proc == 1) {
      r_dest = r_source;
      return;
   }
   // Pointers to indirect the source and dest arrays, if we need to use
   // temporaries
   Array<T,3> * source_pointer, * dest_pointer; 

   assert(all(r_dest.extent() == source_extent) &&
         all(r_dest.lbound() == source_lbound));
   assert(all(r_source.extent() == dest_extent) &&
         all(r_source.lbound() == dest_lbound));

   // See temporary-allocation for the logic of this
   alloc_temps(r_dest, r_source, dest_pointer, source_pointer);

   // Create references, so that the transpoce can use the more familiar
   // array(i,j,k) notation without tons of pointer dereferences cluttering
   // things up.
   Array<T,3> & srcarray = *source_pointer;
   Array<T,3> & dstarray = *dest_pointer;

   // Finally, where's the beef!?
   TinyVector<int,3> send_location, rec_location;
   rec_location(untouched_dim) = send_location(untouched_dim) = 0;
   rec_location(from_split) = source_lbound(from_split);
   send_location(to_split) = dest_lbound(to_split);


   for (int k = 0; k < num_proc; k++ ) {
      /* Since MPI_Alltoallw is not well-supported, we fake it with sendrecv
         calls.  Starting with an offset of 0, each process will send to
         the one (offset) above it [and thus receive from the one (offset)
         below it], wrapping around when we reach 0 / numproc. */
      int sendto = (num_proc + my_rank - k) % num_proc;
      int recfrom = (num_proc + my_rank + k) % num_proc;

      /* From here on, things function exactly as in the forward transpose, save that
         sending and receiving are swapped */
      rec_location(to_split) = base_to[recfrom];
      int rec_count = extent_to[recfrom];
      send_location(from_split) = base_from[sendto];
      int send_count = dest_extent(to_split);

      MPI_Status ignoreme;

      void * sendloc = 0, * recloc = 0;
      if (send_count)
         sendloc = &srcarray(send_location);
      if (rec_count)
         recloc = &dstarray(rec_location);

      // And make the sendrecv call:
     MPI_Sendrecv(sendloc,send_count,rec_types[sendto],sendto,0,
            recloc,rec_count,send_type,recfrom,0,
            comm,&ignoreme);
      
//            printf( "Sending %d partial rows to %d, receiving %d partial rows from %d\n",send_count,sendto,rec_count,recfrom);
   }
   // If we transposed into the temporary, copy back
   if (&dstarray != &r_dest) {
      r_dest = dstarray;
   }
}

template<class T>
void Transposer<T>::source_alloc(TinyVector<int,3> & lbound,
      TinyVector<int,3> & extent, GeneralArrayStorage<3> & order) {
   lbound = source_lbound; extent = source_extent; order = optim_order;
}
   
template<class T>
void Transposer<T>::dest_alloc(TinyVector<int,3> & lbound,
      TinyVector<int,3> & extent, GeneralArrayStorage<3> & order) {
   lbound = dest_lbound; extent = dest_extent; order = optim_order;
}

template class Transposer<int>;
template class Transposer<double>;
template class Transposer<complex<double> >;

