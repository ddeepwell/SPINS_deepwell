#include "../TArray.hpp"
#include "../T_util.hpp"
#include <iostream>
#include <cstdlib>

using namespace TArray;
using blitz::Array;
using blitz::firstIndex;
using blitz::secondIndex;
using blitz::thirdIndex;
using std::atoi;
using std::cout;

int main(int argc, char * argv[]) {
   int szx = 0, szy = 0, szz = 0; /* x, y, z sizes */
   if (argc > 1) { /* single size specified */
      szx = szy = szz = atoi(argv[1]);
   }
   if (argc > 2) { /* Two sizes specified */
      szz = 0;
      szy = atoi(argv[2]);
   }
   if (argc > 3) { /* Three sizes specified */
      szz = atoi(argv[3]);
   }
   if (szx == 0) szx = 10;
   if (szy == 0) szy = 10;
   if (szz == 0) szz = 10;
   DTArray write_testing(szx,szy,szz);
   firstIndex ii; secondIndex jj; thirdIndex kk;
   /* Initialize the array with "coordinates" */
   write_testing = ii + 10*jj + 100*kk;
   /* And test the written outputs */
   write_array(write_testing,"dat_seq",0);
   write_array(write_testing,"dat_seq",1);
   write_reader(write_testing,"dat_seq",true);
   /* Nonsequenced version */
   write_array(write_testing,"dat_nsq");
   write_reader(write_testing,"dat_nsq");
   return 0;
}
   
