#include "../TArray.hpp"
#include "../Parformer.hpp"
#include "../Par_util.hpp"
#include <blitz/array.h>
#include <mpi.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using TArray::DTArray;
using TArray::CTArray;


using blitz::RectDomain;
using blitz::firstIndex;
using blitz::secondIndex;
using blitz::thirdIndex;

using namespace Transformer;


void die() {
   MPI_Finalize();
   exit(1);
}

void test(int myrank, int yesall) {
   if (yesall) {
      if (!myrank) fprintf(stdout, "OK\n");
   } else {
      if (!myrank) fprintf(stdout, "FAILED!\n");
      die();
   }
}



int main(int argc, char ** argv) {
   MPI_Init(&argc, & argv);
   int szx, szy, szz;
   int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   int numproc; MPI_Comm_size(MPI_COMM_WORLD,&numproc);

   firstIndex ii;
   secondIndex jj;
   thirdIndex kk;

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

   DTArray reference_array(szx,szy,szz); // Reference 
   DTArray real_ref(szx,szy,szz);
   CTArray cx_ref(szx/2+1, szy, szz);
   CTArray cy_ref(szx, szy/2+1, szz);
   CTArray cz_ref(szx, szy, szz/2+1);

   DTArray * split_arrayp;
   split_arrayp = alloc_array(szx,szy,szz); // Allocate the split array

   DTArray & split_array = *split_arrayp;

   reference_array = ii + szx*jj + (szx*szy)*kk;
   split_array = ii + szx*jj + (szx*szy)*kk;

   /* First, check that the split array matches the reference array locally */
   if (!myrank)
      fprintf(stdout,"Checking initialization match... ");
   
   double mysum = sum(split_array);
   double allsum;
   MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   test(myrank,(fabs(allsum - sum(reference_array)) < 1e-1));

   if (!myrank)
      fprintf(stdout,"Checking expected failure... ");
   if (!myrank) split_array(0,0,0) = -1;
   mysum = sum(split_array);
   MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   test(myrank,!(fabs(allsum - sum(reference_array)) < 1e-1));
   if (!myrank) split_array(0,0,0) = 0;

   if (!myrank)
      fprintf(stdout, "Testing transforms...\n");

   { /* Test X-transforms */
      if (!myrank) fprintf(stdout,"First Dimension.....\n");
      Trans1D sine(szx,szy,szz,firstDim,SINE), 
              cosine(szx,szy,szz,firstDim,COSINE),
              fourier(szx,szy,szz,firstDim,FOURIER);
      /* sine */
      if (!myrank) fprintf(stdout,"      sine ... ");
      reference_array.transform(real_ref,firstDim,::TArray::DST1);
      sine.forward_transform(&split_array,SINE); 
      mysum = sum(*sine.get_real_temp());
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      test(myrank,fabs(allsum - sum(real_ref)) < 1e-4); 
      if (!myrank) fprintf(stdout,"    cosine ... ");
      reference_array.transform(real_ref,firstDim,::TArray::DCT1);
      cosine.forward_transform(&split_array,COSINE); 
      mysum = sum(*cosine.get_real_temp());
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
      test(myrank,fabs(allsum - sum(real_ref)) < 1e-4); 
      if (!myrank) fprintf(stdout,"   fourier ... ");
      reference_array.transform(cx_ref,firstDim,::TArray::FFTR);
      fourier.forward_transform(&split_array,FOURIER); 
      mysum = sum(abs(*fourier.get_complex_temp()));
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
      test(myrank,fabs(allsum - sum(abs(cx_ref))) < 1e-4); 
   }
   { /* Test X-transforms */
      if (!myrank) fprintf(stdout,"Second Dimension.....\n");
      Trans1D sine(szx,szy,szz,secondDim,SINE), 
              cosine(szx,szy,szz,secondDim,COSINE),
              fourier(szx,szy,szz,secondDim,FOURIER);
      /* sine */
      if (!myrank) fprintf(stdout,"      sine ... ");
      reference_array.transform(real_ref,secondDim,::TArray::DST1);
      sine.forward_transform(&split_array,SINE); 
      mysum = sum(*sine.get_real_temp());
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      test(myrank,fabs(allsum - sum(real_ref)) < 1e-4); 
      if (!myrank) fprintf(stdout,"    cosine ... ");
      reference_array.transform(real_ref,secondDim,::TArray::DCT1);
      cosine.forward_transform(&split_array,COSINE); 
      mysum = sum(*cosine.get_real_temp());
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
      test(myrank,fabs(allsum - sum(real_ref)) < 1e-4); 
      if (!myrank) fprintf(stdout,"   fourier ... ");
      reference_array.transform(cy_ref,secondDim,::TArray::FFTR);
      fourier.forward_transform(&split_array,FOURIER); 
      mysum = sum(abs(*fourier.get_complex_temp()));
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
      test(myrank,fabs(allsum - sum(abs(cy_ref))) < 1e-4); 
   }
   { /* Test Z-transforms */
      if (!myrank) fprintf(stdout,"Third Dimension.....\n");
      Trans1D sine(szx,szy,szz,thirdDim,SINE), 
              cosine(szx,szy,szz,thirdDim,COSINE),
              fourier(szx,szy,szz,thirdDim,FOURIER);
      /* sine */
      if (!myrank) fprintf(stdout,"      sine ... ");
      reference_array.transform(real_ref,thirdDim,::TArray::DST1);
      sine.forward_transform(&split_array,SINE); 
      mysum = sum(*sine.get_real_temp());
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      test(myrank,fabs(allsum - sum(real_ref)) < 1e-4); 
      if (!myrank) fprintf(stdout,"    cosine ... ");
      reference_array.transform(real_ref,thirdDim,::TArray::DCT1);
      cosine.forward_transform(&split_array,COSINE); 
      mysum = sum(*cosine.get_real_temp());
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
      test(myrank,fabs(allsum - sum(real_ref)) < 1e-4); 
      if (!myrank) fprintf(stdout,"   fourier ... ");
      reference_array.transform(cz_ref,thirdDim,::TArray::FFTR);
      fourier.forward_transform(&split_array,FOURIER); 
      mysum = sum(abs(*fourier.get_complex_temp()));
      MPI_Allreduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         
      test(myrank,fabs(allsum - sum(abs(cz_ref))) < 1e-4); 
   }

   MPI_Finalize();
   return 0;
}

   


