#ifndef SPLIT_READER_HPP
#error "split_reader_impl.cc must be included from split_reader.hpp"
#endif

/* Implementation for the 2D split array reader */

#include <sys/mman.h> // Memmap
#include <sys/types.h> // for open
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h> // For close
#include <errno.h> // For errors
#include <stdio.h>
#include <mpi.h>

template <class T>
   blitz::Array<T,2> * read_2d_slice(const char * filename,
            int Nx, int Ny, blitz::Range range_x, blitz::Range range_y,
            blitz::GeneralArrayStorage<2> storage) {
      /* First, open the specified file for reading */
      int my_fd = open(filename,O_RDONLY); // open read only
      if (my_fd == -1) { // error
         fprintf(stderr,"I/O error opening %s\n",filename);
         fprintf(stderr,"The error was: %d, %s\n",errno,strerror(errno));
         exit(1); // and die
      }
      /* Now, map the file into memory, assuming it is of Nx*Ny size
         of type T */
      /* Implementation note -- as a template, this is defined for all sorts
         of classes T that don't make sense written to / read from disk as
         a pure memory copy.  Since blitz arrays are generally numeric in
         nature, this is probably no big deal. */
      T * my_data = (T *) mmap(0,Nx*Ny*sizeof(T), // create proper size
                              PROT_READ, // for read access
                              MAP_SHARED, // shared map
                              my_fd, 0); // at zero offset
      if (my_data == MAP_FAILED) {
         /* Error */
         fprintf(stderr,"Memory map of %s failed\n",filename);
         fprintf(stderr,"The error was: %d, %s\n",errno,strerror(errno));
         exit(1); // and die
      }
      /* Now, my_data contains a pointer to a read-only bunch of memory
         of an on-disk array of size Nx*Ny.  Turn this into a Blitz array. */
      blitz::Array<T,2> disk_array(my_data,blitz::shape(Nx,Ny),
                              blitz::neverDeleteData,storage);

      /* Create our local array of the proper shape and storage order */
      blitz::Array<T,2> * retarray =
               new blitz::Array<T,2>(range_x, range_y, storage);

      /* And copy the right chunk of data from the R/O disk array to the
         R/W memory array */
      *retarray = disk_array(range_x,range_y);

      /* Unmap the memory */
      if (munmap(my_data,sizeof(T)*Nx*Ny) == -1) {
         fprintf(stderr,"Memory unmap of %s failed\n",filename);
         fprintf(stderr,"The error was: %d, %s\n",errno, strerror(errno));
         exit(1);
      }
      /* And close the file descriptor */
      if (close(my_fd) == -1) {
         fprintf(stderr,"Error closing %s\n",filename);
         fprintf(stderr,"The error was: %d, %s\n",errno,strerror(errno));
      } 

      // Place an MPI barrier here to make sure that all processes have
      // read the file before continuing.  This prevents a race condition
      // if the file is supposed to be overwritten later, where a late-
      // to-the-party process might open up the replacement file.
      MPI_Barrier(MPI_COMM_WORLD);

      return retarray;
      
   }
