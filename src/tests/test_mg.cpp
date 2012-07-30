#include "../multigrid.hpp"
#include "../Par_util.hpp"
#include <blitz/array.h>
#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

using namespace blitz;
using namespace std;

firstIndex ii; secondIndex jj;

#define SYNC(__x__) { for (int _i = 0; _i < numproc; _i++) { \
                     if (myrank == _i) { \
                        __x__; \
                     } MPI_Barrier(MPI_COMM_WORLD); } }
#define MASTER (myrank == 0)

#define SIZE_X 1024
#define SIZE_Y 128
//#define SIZE_X 4
//#define SIZE_Y 4

int main(int argc, char ** argv) {
   cout.precision(3);
   cout.width(7);
   cerr.precision(3);
   MPI_Init(&argc, &argv);
   int myrank, numproc;
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   MPI_Comm_size(MPI_COMM_WORLD,&numproc);
   MPI_Barrier(MPI_COMM_WORLD);
   /* Test of multigrid code. */
   /* Build ranges for initializing split arrays.  We'll use 
      4 x-lines per processor */
   int xlbound, xubound;
   get_local_split(SIZE_X,myrank,numproc,xlbound,xubound);
//   SYNC(fprintf(stdout,"Processor %d split: %d to %d\n",myrank,xlbound,xubound));
   Range xrange(xlbound,xubound);
   Range yrange(0,SIZE_Y-1);
   int size_x = SIZE_X;
//   Array<double,1> x_all(size_x+2), xline(xrange), yline(SIZE_Y);
   Array<double,1> x_all(size_x), xline(xrange), yline(SIZE_Y);
   /* Initialize with a Chebyshev grid in x and y */
//   x_all = -cos(M_PI*ii/(size_x-1));
   x_all = (ii-0.5)/double(SIZE_X)*2*M_PI;
//   x_all(0) = 0.5/double(SIZE_X+1);
//   x_all(SIZE_X+1) = x_all(SIZE_X)+0.5/double(SIZE_X+1);
   xline = x_all(xrange);
//   yline = -cos(M_PI*ii/(SIZE_Y-1));
   yline = ii/double(SIZE_Y);
//   yline = -1 + 2.0*ii/(SIZE_Y-1);

   /* Generate split arrays for: */
   Array<double,2> u(xrange,yrange), // solution
      du(xrange,yrange), // Differential
      f(xrange,yrange), // RHS
      resid(xrange,yrange), // Residual
      ones(xrange,yrange), // uniformly one
      zeros(xrange,yrange), // uniformly zero
      nonzeros(xrange,yrange);
   ones = 1; zeros = 0; f = 0; nonzeros = xline(ii)/4+yline(jj)/4;

   /* And local arrays for BCs */
   Array<double,1> zeros_x(xrange), ones_x(xrange), zeros_y(SIZE_Y), ones_y(SIZE_Y);
   zeros_x = 0; zeros_y = 0;
   ones_x = 1; ones_y = 1;

   //f = xline(ii)+0*jj;

   MG_Solver miggy(x_all,yline,SYM_NONE);
   /* Set up the problem u_xx + u_yy = f */
   miggy.problem_setup(ones,ones,zeros,zeros,zeros);
   miggy.helmholtz_setup(0);

   /* And for now, all-dirichlet BCs */
   // u, uz, ux
//   miggy.bc_setup(1,zeros_x,ones_x,zeros_x,zeros_x,ones_x,zeros_x);
//   miggy.bc_setup(0,zeros_y,zeros_y,ones_y,zeros_y,zeros_y,ones_y);
   miggy.bc_setup(1,ones_x,zeros_x,zeros_x,ones_x,zeros_x,zeros_x);
   miggy.bc_setup(0,ones_y,zeros_y,zeros_y,ones_y,zeros_y,zeros_y);

   double max_err;
//         u = exp(cos(xline(ii)-yline(jj))-pow(yline(jj),2));
//         miggy.apply_operator(u,f);
//         resid = f;
//         SYNC(cout << f);

   /* Now, let's set some kind of nontrivial exact solution */
#if 0
   int pre = 1, post = 1;
//   for (int pre = 0; pre <= 3; pre++) 
//      for (int post = 1; post <= 3; post++) 
   {
         u = exp(cos(xline(ii)-yline(jj))-pow(yline(jj),2));
         double function_mean = mean(u);
         double incompat = 0;
//         u = sin(xline(ii)+M_PI/2) + 0*jj;
//         cout << u;
         miggy.apply_operator(u,f);
         u = 0;
         /* Now, let's time convergence */
         double start_norm = sqrt(pssum(sum(pow(f,2))));
         if (pre == 0 && post == 1)
            miggy.cycle(CYCLE_V,resid,du,function_mean,incompat,0,0,0); // Perform one cycle to set up the operator
         double now = MPI_Wtime();
         int count = 0;
         double current_norm;
         incompat = 0;
         double dmean = function_mean;
         resid = f;
         do {
            miggy.cycle(CYCLE_V,resid,du,dmean,incompat,pre,0,post);
//            cout << du;
            u = u + du;
            miggy.apply_operator(u,resid);
            resid = f - resid - incompat;
            dmean = function_mean - mean(u);
//            cout << resid;
//            cout << u;
            count++;
            current_norm = sqrt(pssum(sum(pow(resid,2))));
            if (master()) fprintf(stdout,"%d: %g + %g + %g\n",count,current_norm/start_norm,incompat,dmean);
         } while (current_norm > 1e-6*start_norm && count <= 30);
         double later = MPI_Wtime();
         if (master()) {
            if (current_norm > 1e-6*start_norm)
               fprintf(stdout,"V(%d,%d)-cycle time (-- x %.2g): %f seconds\n",pre,post,current_norm/start_norm,later-now);
            else
               fprintf(stdout,"V(%d,%d)-cycle time (%d): %f seconds\n",pre,post,count,later-now);
         }
   }
#endif
#if 1
   for (int pre = 0; pre <= 2; pre++) 
      for (int mid = 0; mid <= 3; mid++)
         for (int post = 0; post <= 3; post++) 
//   int pre = 1, mid = 1, post = 1;
   {
      //u = exp(cos(xline(ii)-yline(jj))-pow(yline(jj),2));
//      u = sin(ii)+cos(jj)+sin(ii+jj);
      u = exp(cos(xline(ii)-yline(jj)))*cos(yline(jj)+xline(ii));
         double function_mean = pvsum(u)/(SIZE_X*SIZE_Y);
         double incompat = 0;
//      u = pow(xline(ii)*yline(jj),2);
//      u = xline(ii)*yline(jj);
//            u = xline(ii)*xline(ii)+yline(jj)*yline(jj);
      miggy.apply_operator(u,f);
      f = f;
      resid = -f;
      f = 0;

//      SYNC(cout << f);
//      MPI_Finalize(); exit(1);
      double start_norm = sqrt(pssum(sum(pow(f,2))));
      start_norm = 1;
//      resid = resid / start_norm; start_norm = 1;
//      u = 0;
      /* Now, let's time convergence */
      double now = MPI_Wtime();
      int count = 0;
      double current_norm;
         incompat = 0;
         double dmean = function_mean;
      do {
         double dincompat = 0;
        miggy.apply_operator(u,resid);
        resid = resid + incompat;
         dmean = pvsum(u)/(SIZE_X*SIZE_Y);
//         miggy.cycle(CYCLE_V,resid,du,dmean,dincompat,pre,mid,post);
//         du = resid / -1;
         miggy.cycle(CYCLE_F,resid,du,dmean,dincompat,pre,mid,post);
//         SYNC(cout<<du);
//         u = du;
         u = du - u;
         incompat = dincompat - incompat;
//         cout << u;
         current_norm = sqrt(pssum(sum(pow(u,2))))/(SIZE_X*SIZE_Y);
         u = u / current_norm;
         if (pvsum(u) < 0) {
            u = -u;
         }
//            dmean = function_mean - pvsum(u)/(SIZE_X*SIZE_Y);
//            SYNC(cout << resid);
         count++;
//         if (master()) fprintf(stderr,"%d: %g + %g + %g\n",count,current_norm,incompat,dmean);

//         if (master()) fprintf(stdout,"%d: %g + %g + %g\n",count,current_norm/start_norm,incompat,dmean);
//         sleep(1);
      } while (count < 25);
      double later = MPI_Wtime();
      if (master()) {
         if (current_norm < 0.5)
            fprintf(stderr,"F(%d,%d,%d): %.2g (%.2gs per, %.2gs/digit)\n",pre,mid,post,current_norm,(later-now)/count,-(later-now)/count/(log(current_norm)/log(10)));
         else 
            fprintf(stderr,"F(%d,%d,%d): %.2g (%.2gs per, --)\n",pre,mid,post,current_norm,(later-now)/count);

      }
   }
//   miggy.apply_operator(u,du);
//   du = du - f;

//      cout << du;

//   cout << f << u;
#endif
   MPI_Finalize();
   return 0;
}
