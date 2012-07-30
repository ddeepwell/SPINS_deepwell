#include "../Splits.hpp"
#include "../TArray.hpp"
#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

using TArray::DTArray;
using TArray::Trans;
using TArray::DCT1;

using TArray::Dimension;



/* Timing test for an MPI-split, 3D array, taking a cosine transform in each
   dimension.  The main question to answer is whether the parllization is going
   to be a significant performance impact for large arrays. */

int main(int argc, char ** argv) {
   MPI_Init(&argc, & argv);
   int szx, szy, szz;
   int myrank; MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   int numproc; MPI_Comm_size(MPI_COMM_WORLD,&numproc);
   if (argc < 4) {
      if (!myrank) fprintf(stderr,"Not enough input arguments\n");
      MPI_Finalize(); return 1;
   }
   szx = atoi(argv[1]);
   szy = atoi(argv[2]);
   szz = atoi(argv[3]);
   if (szx <= 0 || szy <= 0 || szz <= 0) {
      if (!myrank) fprintf(stderr,"Invalid sizes specified\n");
      MPI_Finalize(); return 1;
   }
   Transposer<double> trans(szx,szy,szz,Dimension(firstDim),Dimension(thirdDim),MPI_COMM_WORLD,MPI_DOUBLE);
   TinyVector<int,3> lbound, extent; GeneralArrayStorage<3> order;
   trans.source_alloc(lbound, extent, order);
   DTArray starting_array(lbound,extent,order);
   DTArray temp1_array(lbound,extent,order);
   DTArray temp2_array(lbound,extent,order);
   trans.dest_alloc(lbound,extent,order);
   DTArray transposed_array(lbound,extent,order);
   DTArray temp3_array(lbound,extent,order);
   

   starting_array = 0;

   /* Transform local dimensions */
   int jjj; double now;
   MPI_Barrier(MPI_COMM_WORLD);
   for (jjj=-1; jjj < 100; jjj++) {
      if (jjj == 0) {
         MPI_Barrier(MPI_COMM_WORLD);
         now = MPI_Wtime();
      }
      starting_array.transform(temp1_array,Dimension(secondDim),DCT1);
      temp2_array.transform(temp2_array,Dimension(thirdDim),DCT1);
      if (numproc > 1) {
/*         if (jjj == -1) 
            if (!myrank)
               fprintf(stderr,"Transposing arrays!\n");*/
         trans.transpose(temp2_array,transposed_array);
         transposed_array.transform(temp3_array,Dimension(firstDim),DCT1);
         trans.back_transpose(temp3_array,starting_array);
      } else {
         temp2_array.transform(temp3_array,Dimension(firstDim),DCT1);
      }
   }
   double later = MPI_Wtime();
   double bytes = double(sizeof(double))*szx*szy*szz*jjj;
   if (!myrank) {
      fprintf(stdout,"%g MiB mangled in %g sec, for %g MiB/sec throughput\n",
            bytes/1024/1024,later-now,bytes/1024/1024/(later-now));
   }
   MPI_Finalize();
   return 0;
}
