/* FluidEvolve.hpp -- control class for Navier-Stokes (incompressible) 
   integration */
#ifndef NSINTEGRATOR_HPP
#define NSINTEGRATOR_HPP 1

/* FluidEvolve is the "physical guts" of SPINS, in the way that ESolver is 
   the "numerical guts".  FluidEvolve controls:
      1) The maximum allowable timestep, via CFL conditions.
      2) Projection of velocities
      3) Computation of pressure (via ESolver)
      4) Constructing Helmholtz solutions for viscosity (via ESolver)
      5) Stepping any tracers (to be implemented)
   
   It's all that and a bag of chips.  It's even a template class, to allow
   for plug-in user control code.  This code is responsible for:
      1) Setting the maximum timestep, for cosmetic or plotting purposes
         (That is, user code may choose a timestep lower than what FluidEvolve
         gives it, if it would result in desirable timelevels being hit
         exactly.)
      2) Velocity forcing, /including/ gravitational and rotational forces.
         (Why? Because especially gravitational forces depend on exactly how
         you specify density.  With a Boussinesq approximation, it's common
         to have a "background" stratification, which shows up in a forcing
         term in the density equation.)
      3) All user-readable IO, including writing fields and any scalar
         diagnostics.  Indeed, what is "interesting" here varies so much
         from problem to problem that this is the original motivation for
         spinning control code into a user-supplied class. */
#include <vector>
#include <set>
#include "ESolver.hpp"
#include "Timestep.hpp"
#include "TArray.hpp"
#include "T_util.hpp"
#include "Parformer.hpp"
#include <stdio.h>
#include "Par_util.hpp"
#include "grad.hpp"

// Global values to control filtering
extern double f_cutoff, f_order, f_strength;
extern bool zero_tracer_boundary;
namespace NSIntegrator {
   using namespace TArrayn;
   using namespace Timestep;
   using namespace ESolver;
   using std::vector;
   using std::set;
   using namespace Transformer;
   // List supported physical dimension types (Fourier, free slip, no slip, mapped)  
   enum DIMTYPE { 
      PERIODIC, 
      FREE_SLIP, // Cosine expansion
      NO_SLIP    // Chebyshev expansion
   };
   enum Sym { // Symmetry
      EVEN, ODD };
   template <class Control> class FluidEvolve {
      public:
         /* This will have to include local/global ranges for parallelization*/
         int szx, szy, szz; // Sizes

         DIMTYPE tx, ty, tz; // Dimension types

         /* This will have to include ranges (arrays) for topography */
         double Lx, Ly, Lz; // domain lengths
         double visco;

         Control * usercode;

         /* Parameters for allocating arrays, given parallel splitting */
         blitz::TinyVector<int,3> local_lbounds, local_extents;
         blitz::GeneralArrayStorage<3> local_storage;

         /* Arrays for velocities and their previous timelevel forcings */
         Stepped<DTArray> us, us_forcing, vs, vs_forcing, ws, ws_forcing;

         /* Arrays for tracers.  Since we have a variable number of them,
            we'll bind the Steppeds up in a vector. */
         vector<Stepped<DTArray> *> tracers, tracers_forcing;

         /* Since we can't exactly pass arbitary numbers of arrays to
            usercode directly, wrap them up in a vector */
         vector<DTArray *> tracers_now, tracers_forcing_now;

         /* And for tracers with diffusion */
         vector<double> t_diffusivity;

         /* Tracer boundary conditions */
         vector<double> t_xbc_dir, t_xbc_neu, // x-boundary condition, dirichlet/neumann
                        t_ybc_dir, t_ybc_neu,
                        t_zbc_dir, t_zbc_neu;

         // Reduced-dimension BCs for boundary condition storage
         Array<double,2> left_bc, right_bc, top_bc, bottom_bc;

         /* Arrays for control of timestepping */
         Stepped<double> times, lhs_coefs, rhs_coefs;

         set<DTArray *> used_temps; // Temporary arrays
         vector<DTArray *> unused_temps;
         
         DTArray pressure; // Pressure is not timestepped

         ElipSolver * pres_solver; // Elliptic solvers
         ElipSolver * visc_solver; 

         /* 3D transforms for filtering and solving */
         TransWrapper * solver_spec, *filter_spec;

         /* Instead of multiple Trans1Ds, use a single grad
            object.  This accounts for Jacobian transforms
            as well */
         Grad * gradient;
//         Trans1D *trans_x, *trans_y, *trans_z; // 1D transforms

         /* Transform type specifiers */
         S_EXP Sx, Sy, Sz;

         // For a mapped domain, we also need to keep track of the coordinate-
         // transform, namely its Jacobian.  The details for derivatives get
         // wrapped up in the Grad above, but boundary conditions (pressure!)
         // and timestep-checking require more direct vector-math.

         // Since only 2D mapping is implemented for now, consider the computational
         // "box" coordintes as alpha and beta.  We want to keep x_alpha, x_beta,
         // z_alpha, and z_beta.

         bool mapped_problem;
         DTArray * x_alpha, * x_beta, * z_alpha, * z_beta;

         /* Constructor, initializing lots 'o stuff */
         /* C++ note: initialization here is done in -declaration order- above,
            rather than order specified in this list.  It's already bitten
            me once, so I hope not to forget it. */
         FluidEvolve(Control * user):
            /* Sizes and scalar data */
            szx(user->size_x()), szy(user->size_y()), szz(user->size_z()),
            //Expansion types
            tx(user->type_x()), ty(user->type_y()), tz(user->type_z()),
            /* Lengths */
            Lx(user->length_x()), Ly(user->length_y()), Lz(user->length_z()),
            /* Viscosity */
            visco(user->get_visco()),
            /* User code */
            usercode(user),
            /* Parallel allocation */
            local_lbounds(alloc_lbound(szx,szy,szz)),
            local_extents(alloc_extent(szx,szy,szz)),
            local_storage(alloc_storage(szx,szy,szz)),
            /* length 4 Stepped variables */
            us(4), us_forcing(4), vs(4), vs_forcing(4), ws(4), ws_forcing(4),
            /* Tracers */
            tracers(user->numtracers()), tracers_forcing(tracers.size()),
            tracers_now(tracers.size()), tracers_forcing_now(tracers.size()),
            t_diffusivity(tracers.size()),
            /* Tracer boundary conditions */
            t_xbc_dir(tracers.size()), t_xbc_neu(tracers.size()),
            t_ybc_dir(tracers.size()), t_ybc_neu(tracers.size()),
            t_zbc_dir(tracers.size()), t_zbc_neu(tracers.size()),
            // Tracer boundary condition arrays
            left_bc(szy,szz), right_bc(szy,szz),
            top_bc(split_range(szx),blitz::Range(0,szy-1)), 
            bottom_bc(split_range(szx),blitz::Range(0,szy-1)),
            /* Timestep coefficients */
            times(4), lhs_coefs(4), rhs_coefs(4),
            /* Misc */
            used_temps(), unused_temps(),
            /* Pressure */
            pressure(local_lbounds,local_extents,local_storage),
            mapped_problem(user->is_mapped()), x_alpha(0), x_beta(0), z_alpha(0),
            z_beta(0) {
               /* For now, only the vertical can be no-slip */
               assert (ty == PERIODIC || ty == FREE_SLIP);
               /* Initialize Stepped's in tracers */
               for (unsigned int i = 0; i < tracers.size(); i++) {
                  tracers[i] = new Stepped<DTArray>(4);
                  tracers_forcing[i] = new Stepped<DTArray>(4);
                  t_diffusivity[i] = user->get_diffusivity(i);
                  // Also get tracer boundary conditions
                  user->tracer_bc_x(i,t_xbc_dir[i],t_xbc_neu[i]);
                  user->tracer_bc_y(i,t_ybc_dir[i],t_ybc_neu[i]);
                  user->tracer_bc_z(i,t_zbc_dir[i],t_zbc_neu[i]);
                  for (int j = -2; j <= 1; j++) {
                     tracers[i]->setp(j,alloc_array());
                     tracers_forcing[i]->setp(j,alloc_array());
                  }
               }

               /* Initialize all of the timestepped-arrays */
               for (int i = -2; i <= 1; i++) {
                  us.setp(i,alloc_array());
                  vs.setp(i,alloc_array());
                  ws.setp(i,alloc_array());
                  us_forcing.setp(i,alloc_array());
                  vs_forcing.setp(i,alloc_array());
                  ws_forcing.setp(i,alloc_array());
               }
               /* More cleanly determine transform types for the elliptic
                  solver and base lengths for Jacboian setting */
               double bl_x, bl_y, bl_z;
               switch (tx) {
                  case PERIODIC: Sx = FOURIER; bl_x=2*M_PI; break;
                  case FREE_SLIP: Sx = REAL; bl_x = M_PI; break;
                  case NO_SLIP: 
                                  Sx = CHEBY; 
                                  bl_x = -2;
                                  if (Lx < 0) {
                                    bl_x = 2; Lx = -Lx;
                                  } 
                                  break;
                  default: abort(); break;
               } switch (ty) {
                  case PERIODIC: Sy = FOURIER; bl_y = 2*M_PI; break;
                  case FREE_SLIP: Sy = REAL; bl_y = M_PI; break;
                  default: abort(); break;
               } switch (tz) {
                  case PERIODIC: Sz = FOURIER; bl_z = 2*M_PI; break;
                  case FREE_SLIP: Sz = REAL; bl_z = M_PI; break;
                  case NO_SLIP: 
                                  Sz = CHEBY; 
                                  bl_z = -2;
                                  if (Lz < 0) {
                                    bl_z = 2; Lz = -Lz;
                                  } 
                                  break;
                  default: abort();
               }
               if (master()) {
                  fprintf(stderr,"Beginning Navier-Stokes timestepping, on a %d x %d x %d grid\n",szx,szy,szz);
                  fprintf(stderr, "X-dimension: type %d, expansion %d\n",tx,Sx);
                  fprintf(stderr, "Y-dimension: type %d, expansion %d\n",ty,Sy);
                  fprintf(stderr, "Z-dimension: type %d, expansion %d\n",tz,Sz);
                  fprintf(stderr, "%d tracers\n",(int) tracers.size());
                  for (unsigned int k = 0; k < tracers.size(); k++) {
                     if (t_diffusivity[k] == 0) {
                        fprintf(stderr,"   tracer %d nondiffusive\n",k);
                     } else
                        fprintf(stderr,"   tracer %d diffusivity %g\n",k,t_diffusivity[k]);
                  }
                  if (visco == 0) {
                     fprintf(stderr,"Inviscid problem\n");
                  } else {
                     fprintf(stderr,"Viscosity %g\n",visco);
                  }
                  if (mapped_problem) {
                     fprintf(stderr,"Mapped grid\n");
                  } else {
                     fprintf(stderr,"Unmapped grid\n");
                  }
               }

               /* 1D line transformers for forward derivatives */
//               trans_x = new Trans1D(szx,szy,szz,firstDim, Sx);
//               trans_y = new Trans1D(szx,szy,szz,secondDim, Sy);
//               trans_z = new Trans1D(szx,szy,szz,thirdDim, Sz);
               /* Use gradient */
               gradient = new TArrayn::Grad(szx,szy,szz,Sx,Sy,Sz);
               user->set_grad(gradient);
               /* Now, set the Jacobian properly.  In lieu of modifying
                  BaseCase to return a Jacobian directly (necessary for
                  mapping), set this automatically based on lengths */
               if (mapped_problem) {
                  if (master()) fprintf(stderr,"Performing mapping\n");
                  assert (Sz == CHEBY);
                  x_alpha = new DTArray(local_lbounds,local_extents,local_storage);
                  x_beta = new DTArray(local_lbounds,local_extents,local_storage);
                  z_alpha = new DTArray(local_lbounds,local_extents,local_storage);
                  z_beta = new DTArray(local_lbounds,local_extents,local_storage);
                  gradient->set_jac(firstDim,firstDim,1);
                  gradient->set_jac(secondDim,secondDim,1);
                  gradient->set_jac(thirdDim,thirdDim,1);
                  DTArray & xgrid = *get_temp();
                  DTArray & ygrid = *get_temp();
                  DTArray & zgrid = *get_temp();
                  if (master()) fprintf(stderr,"Getting mapped coordinates\n");
                  usercode->do_mapping(xgrid,ygrid,zgrid);
                  // Remove the base length in x from xgrid,
                  // because a linear map (x->x) doesn't play well with a trig
                  // expansion.
                  if (Sx == SINE || Sx == COSINE || Sx == FOURIER || Sx==REAL) {
                     blitz::firstIndex ii;
                     xgrid = xgrid - Lx*(0.5+ii)/szx;
                  }
                  if (Sx == CHEBY) {
                     blitz::firstIndex ii;
                     xgrid = xgrid - (Lx/bl_x)*cos(ii*M_PI/(szx-1));
                  }
                  if (master()) fprintf(stderr,"Computing Jacobian\n");
                  gradient->setup_array(&xgrid,(Sx==SINE || Sx==REAL ? COSINE : Sx),Sy,Sz);
                  gradient->get_dx(x_alpha);
                  *x_alpha = *x_alpha + Lx/bl_x;
                  gradient->get_dz(x_beta);
                  gradient->setup_array(&zgrid,(Sx==SINE || Sx==REAL ? COSINE : Sx),Sy,Sz);
                  gradient->get_dx(z_alpha);
                  gradient->get_dz(z_beta);
//                  cerr << *x_alpha << *x_beta << *z_alpha << *z_beta;
//                  exit(1);

                  // Reuse ygrid as temporary, since mapping is strictly 2D for now

                  // For chain-rule application of derivatives, need terms in 
                  // dalpha/dx form rather than vice versa.  Invert the 2x2 matrix.
                  ygrid = *z_beta/((*x_alpha)*(*z_beta)-(*x_beta)*(*z_alpha));
                  gradient->set_jac(firstDim,firstDim,0,&ygrid);
                  ygrid = -*z_alpha/((*x_alpha)*(*z_beta)-(*x_beta)*(*z_alpha));
                  gradient->set_jac(firstDim,thirdDim,0,&ygrid);
                  ygrid = -*x_beta/((*x_alpha)*(*z_beta)-(*x_beta)*(*z_alpha));
                  gradient->set_jac(thirdDim,firstDim,0,&ygrid);
                  ygrid = *x_alpha/((*x_alpha)*(*z_beta)-(*x_beta)*(*z_alpha));
                  gradient->set_jac(thirdDim,thirdDim,0,&ygrid);

                  gradient->set_jac(secondDim,secondDim,bl_y/Ly);
                     
               } else {
                  gradient->set_jac(firstDim,firstDim,bl_x/Lx);
                  gradient->set_jac(secondDim,secondDim,bl_y/Ly);
                  gradient->set_jac(thirdDim,thirdDim,bl_z/Lz);
               }

               filter_spec = new TransWrapper(szx,szy,szz,Sx,Sy,Sz);
               /*No slip boundaries (Chebyshev discretization)
                 mean that a dimension doesn't reduce to a pure algebraic
                 division to invert the Laplacian.  Therefore, the
                 tansformer that we pass to the solver must have type
                 NONE in the vertical (and possibly horizontal). */
               solver_spec = new TransWrapper(szx,szy,szz,(Sx==CHEBY ? NONE : Sx),Sy,
                     (Sz==CHEBY) ? NONE : Sz);
                  
               pres_solver = new ElipSolver(0, solver_spec, gradient);
               visc_solver = pres_solver;
            }

         /* Destructor -- deallocate temporary arrays */
         ~FluidEvolve() {
            for (set<DTArray*>::iterator i = used_temps.begin(); 
                  i != used_temps.end(); i++) {
               delete *i;
            } for (unsigned int i = 0; i < unused_temps.size(); i++) {
               delete unused_temps[i];
            }
            delete pres_solver;
            delete filter_spec;
            delete solver_spec;
            delete gradient;
//            delete trans_x; delete trans_y; delete trans_z;
            /* Destroy tracers */
            for (unsigned int i = 0; i < tracers.size(); i++) {
               delete tracers[i];
               delete tracers_forcing[i];
            }
         }

         void initialize();
         void do_step(); // Do one timestep, given levels in times[]
         void do_run(double fintime); // Run many timesteps
         

      private:
         /* Manage temporary arrays */
         DTArray * get_temp(); // Get a temporary array, allocate if necessary
         void release_temp(DTArray *); // Release a temporary array

         DTArray * alloc_array(); // Allocate a full-sized array

         /* Compute divergence of projected velocities */
         void divergence(DTArray & up, DTArray & vp, DTArray & wp, 
                        DTArray & div, Sym sx, Sym sy, Sym sz);
         /* Gradient, used for advection and pressure effects */
         void grad(DTArray & vec, DTArray & dx, DTArray & dy, DTArray & dz,
               Sym sx, Sym sy, Sym sz);

         /* Appication of timestep coefs to an array, for computing projected
            velocities */

         void apply_timestep(Array<double, 3> & ar, Stepped<DTArray> & lhs,
                             Stepped<DTArray> & rhs);

         double get_cfl(); // Get maximum allowable timestep from CFL condition

         /* Shift all of the stepped arrays, to clear foo[1] for use at the
            next timestep */
         void shift_arrays(); 

   };
   
   /* Helper function for transform typing */
   S_EXP get_trans_type(DIMTYPE dtype, Sym stype);
}

#include "NSIntegrator_impl.cc"



#endif // NSINTEGRATOR_HPP
