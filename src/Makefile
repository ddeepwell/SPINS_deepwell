# Makefile to build SPINS, generalized for multiple systemd

# The "system.mk" file in the current directory will contain
# system-specific make variables, most notably the
# C/C++ compiler/linker, 
include system.mk


# Compile with debugging flags
DEBUG?=false
# Compile with optimizations
OPTIM?=true
# Compile with extra optimizations that may require significantly
# longer for compilation
SLOW_OPTIM?=false

# Assemble the CFLAGS
CFLAGS:=$(SYSTEM_CFLAGS)
LDFLAGS:=$(SYSTEM_LDFLAGS)
ifeq ($(DEBUG),true)
   CFLAGS:=$(CFLAGS) $(DEBUG_CFLAGS)
   LDFLAGS:=$(LDFLAGS) $(DEBUG_LDFLAGS)
endif
ifeq ($(OPTIM),true)
   CFLAGS:=$(CFLAGS) $(OPTIM_CFLAGS)
   LDFLAGS:=$(LDFLAGS) $(OPTIM_LDFLAGS)
   ifeq ($(SLOW_OPTIM),true)
      CFLAGS:=$(CFLAGS) $(EXTRA_OPTIM_CFLAGS)
      LDFLAGS:=$(LDFLAGS) $(EXTRA_OPTIM_LDFLAGS)
   endif
endif

INCLUDE_DIRS := $(MPI_INCDIR) $(LAPACK_INCDIR) $(BLITZ_INCDIR) $(FFTW_INCDIR) $(UMF_INCDIR) -I../include
CFLAGS := $(CFLAGS) $(INCLUDE_DIRS)

LIB_DIRS := $(MPI_LIBDIR) $(LAPACK_LIBDIR) $(BLITZ_LIBDIR) $(FFTW_LIBDIR) $(UMF_LIBDIR) -L../lib
LDLIBS := $(MPI_LIB) $(LAPACK_LIB) -lfftw3 -lblitz -lm -lstdc++ -lumfpack -lamd
LDFLAGS := $(LDFLAGS) $(LIB_DIRS) 

.PHONY: all 
all: tests/test_deriv_x tests/test_write_x tests/test_esolve_x tests/test_heat_x tests/test_ode_x tests/test_ns_x

.PHONY: clean
clean:
	rm -f *.o tests/*.o cases/*.o

objfiles: $(shell ls *.cpp | sed -e 's/cpp/o/g')

NSIntegrator.o: NSIntegrator.cpp NSIntegrator_impl.cc

tests/test%.o: tests/tests%.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

tests/test%_x: tests/test%.o TArray.o Parformer.o T_util.o  ESolver.o Timestep.o NSIntegrator.o BaseCase.o Science.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS) 

cases/%.o: cases/%.cpp NSIntegrator_impl.cc NSIntegrator.hpp
	$(CXX) $(CFLAGS) -o $@ -c  $<


nonhydro_x: nonhydro_sw.o TArray.o T_util.o Parformer.o Splits.o Par_util.o Split_reader.o 
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

derek_x: derek.o TArray.o T_util.o Parformer.o ESolver.o Timestep.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o 
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

cases/%.x: cases/%.o TArray.o T_util.o Parformer.o ESolver.o Timestep.o NSIntegrator.o BaseCase.o Science.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

cases/%_x: cases/%.o TArray.o T_util.o Parformer.o ESolver.o Timestep.o NSIntegrator.o BaseCase.o Science.o Splits.o Par_util.o Split_reader.o gmres.o gmres_1d_solver.o gmres_2d_solver.o grad.o multigrid.o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o : %.cpp	*.hpp
	$(CXX) $(CFLAGS) -o $@ -c $< 

