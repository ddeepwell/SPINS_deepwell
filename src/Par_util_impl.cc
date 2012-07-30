#ifndef PAR_UTIL_HPP
   #error par_util_impl.cc must be included from within par_util.hpp
#endif
#ifndef PAR_UTIL_IMPL
#define PAR_UTIL_IMPL 1
// max
template <class T> T psmax(T x, MPI_Comm c , MPI_Datatype type ) {
   T retval;
   MPI_Allreduce(&x, &retval, 1, type, MPI_MAX, c);
   return retval;
} 
// min
template <class T> T psmin(T x, MPI_Comm c , MPI_Datatype type ) {
   T retval;
   MPI_Allreduce(&x, &retval, 1, type, MPI_MIN, c);
   return retval;
}
// sum
template <class T> T pssum(T x, MPI_Comm c , MPI_Datatype type ) {
   T retval;
   MPI_Allreduce(&x, &retval, 1, type, MPI_SUM, c);
   return retval;
}
// any
template <class T> bool psany(T x, MPI_Comm c ) {
   int local = int(bool(x));
   int retval;
   MPI_Allreduce(&local, &retval, 1, MPI_INT, MPI_LOR, c);
   return bool(retval);
}
// all
template <class T> bool psall(T x, MPI_Comm c){
   int retval;
   int local = int(bool(x));
   MPI_Allreduce(&local, &retval, 1, MPI_INT, MPI_LAND, c);
   return bool(retval);
}

/* Vector operations */
template <class T, int d> T pvmax(const blitz::Array<T,d> & x, 
      MPI_Comm c , MPI_Datatype type ) {
   T local = max(x);
   return psmax(local,c,type);
}
template <class T, int d> T pvmin(const blitz::Array<T,d> & x, 
      MPI_Comm c , MPI_Datatype type ) {
   T local = min(x);
   return psmin(local,c,type);
}
template <class T, int d> T pvsum(const blitz::Array<T,d> & x, 
      MPI_Comm c , MPI_Datatype type) {
   T local = sum(x);
   return pssum(local,c,type);
}
#endif
