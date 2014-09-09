#include <stdio.h>

/* Write the source code of the case to spinscase.cpp */
void WriteCaseFileSource(void)
{
   FILE* fid;
   fid=fopen("spinscase.cpp","w");
   fprintf(fid,"/* %s */\n%s",casefilename,casefilesource);
   fclose(fid);
}
