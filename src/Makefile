CC = mpicc
CXX = mpicc
LD = mpicc
# -ip -ipo seem to give about 10% better speed, but they also double compile
# times.  For now, disable them.
BLITZ_DEBUG =   -g   -DBZ_DEBUG #-ftrapuv -check-uninit
SLOW_OPTIM =   -O3 -ffast-math  #-ip -ipo 
# Compile with all warnings, but disable some remarks.  Notably:
# 1572-- "unreliable" == comparison of floating points
# 981 -- unspecified order of evaluation of operands
# 444 -- destructor is not virtual
# 383 -- value copied to temporary, reference to temporary used
# 869 -- parameter not referenced
INCLUDE_DIRS = -I../UMFPACK/Include -I../AMD/Include  -I../UFconfig
CFLAGS =  $(INCLUDE_DIRS) $(BLITZ_DEBUG) $(SLOW_OPTIM) #-Wall -wd981 -wd444 -wd1572 -wd383 -wd869
# disabling notifications for
# 11000 - performing multi-file optimizations
# 11005 - generating object file <temporary name>.o
LIB_DIRS = -L../UMFPACK/Lib -L../AMD/Lib #-L/opt/scsl/lib
LAPACK_LIB = -llapack
UMFPACK_LIB = -lumfpack -lamd
LDFLAGS =  $(SLOW_OPTIM)  #-wd11000,11005
LDLIBS = $(LIB_DIRS) -lfftw3 -lblitz -lm -lstdc++ -lmpi $(LAPACK_LIB) $(UMFPACK_LIB)

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

