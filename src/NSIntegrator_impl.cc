// CPP file for NSIntegrator.hpp.  Because this is templated, it must
// be included in NSIntegrator.hpp proper.
#ifdef NSINTEGRATOR_HPP
#include "Par_util.hpp"
namespace NSIntegrator {
   template <class C>
      void FluidEvolve<C>::initialize() {
         /* Initial values are taken from user code. Forcing is 0 to
            begin with. */
         times[0] = usercode->init_time();
         // Since we're taking low-order steps to start, copy our initial
         // timestep to previous entries
         times[-2] = times[-1] = times[0];
         usercode->init_vels(us[0], vs[0], ws[0]);

         /* Initialize tracers */
         for (unsigned int i = 0; i < tracers.size(); i++) {
            tracers_now[i] = &(*tracers[i])[0];
         }
         usercode->init_tracers(tracers_now);

         // Tracer generation/initialization will go here
      }
   template <class C>
      DTArray * FluidEvolve<C>::alloc_array() {
         /* Allocate a new DTArray, properly split for parallelization.
            In the constructor, we store the proper lbouds/extents/
            storage-order, so we can use them here.  This more or less
            duplicates the alloc_array function in Par_util.hpp, but
            c'est la vie. */
         return new DTArray(local_lbounds,local_extents,local_storage);
      }
   template <class C>
      void FluidEvolve<C>::do_run(double fintime) {
         /* Control framework for a full run.  At the first timestep(s), take
            fractional timesteps to start up. */
      
         while (times[0] < (fintime - 1e-8*fabs(fintime))) {
            // Since we're not done, we need to take a timestep.

            // Find the desired timestep:
            double cfl = get_cfl();
            //fprintf(stderr,"Got CFL condition of %f\n",cfl);
            double last_step = times[0] - times[-1];
            double my_timestep;
            if (last_step == 0) last_step = cfl;
            /* There's little need to change timestep size if it's close
               to the maximum. */
            if ((last_step > 0.8*cfl) && (last_step < cfl))
               my_timestep =  last_step;
            else my_timestep = fmin(cfl,1.5*last_step);
            if (my_timestep > ((1+1e-8)*(fintime - times[0])))
               my_timestep = (1+5e-9)*(fintime - times[0]);

            
            
            /* Let the usercode override this timestep.  Hitting (or nearly
               hitting, subject to rounding error) a timestep may be important
               for plotting purposes or for physics */
            double the_timestep = usercode->check_timestep(my_timestep,
                                                           times[0]);

            /* If the user code returns a nonpositive timestep, take it as
               an abort signal.  Stepping backwards makes no sense. */
            if (the_timestep <= 0) {
               if (master())
                  fprintf(stderr,"Timestep of %g is too low to continue, aborting\n",the_timestep);
               return;
            }

            double desttime = times[0] + the_timestep;

            /* Now, take timesteps until we reach our one-timestep destination*/
            
            bool starting_step = false;
            double my_step = 0;
            while (times[0] < (desttime - 1e-10*fabs(desttime))) {
               /* Why bundle this up in a loop?  Starting timesteps.  For the
                  first few timesteps, we don't have a full history of prior
                  timelevels, so we have to take an Euler (or AB2) step.

                  These schemes are less accurate, and if we took a full step
                  of that size the /one-step/ inaccuracy would dominate the
                  total time integration error.  So, if we're starting up,
                  we take a small timestep and progressively double it. */
               if (times[-1] == times[-2]) {
                  /* Startup, so take a tiny timestep */
                  my_step = the_timestep/64;
                  starting_step = true;
                  //fprintf(stderr,"Starting step, size: %g, t=%g\n",my_step,times[0]);
               } else if (starting_step) {
                  /* We've finished the low-level startup, so we want
                     to increase our step size back to normal.  Doubling
                     the timestep each of these substeps works, as:
                     the_timestep(1/16 + 1/16 + 1/8 + 1/4 + 1/2)
                     is an entire timestep. */
                  my_step = my_step * 2;
                  //fprintf(stderr,"Continuing step, size: %g, t=%g\n",my_step,times[0]);
               } else { // Not starting
                  my_step = the_timestep;
                  //fprintf(stderr,"Full step, size: %g, t=%g\n",my_step,times[0]);
               }

               /* Take a single timestep */
               times[1] = times[0] + my_step;
               do_step();

               shift_arrays();
               // Talk to usercode for any plotting
               /* First, set current tracers */
               for (unsigned int i = 0; i < tracers.size(); i++) {
                  tracers_now[i] = &((*tracers[i])[0]);
               }
               usercode->analysis(times[0],us[0],vs[0],ws[0],tracers_now,pressure);
            }
         }
         return;
      }
               
         
   template <class C> 
      void FluidEvolve<C>::do_step() {
         /* Evolve fields in us[-2->0] (etc) to timelevel in times[1] */

         // First, get temporary arrays for gradients, divergence, (U.U)
         DTArray * restrict dx = get_temp();
         DTArray * restrict dy = get_temp();
         DTArray * restrict dz = get_temp();
         DTArray * restrict sca_temp = get_temp(); // scalar temporary

         // Projected velocities will go in us_forcing/etc.

         // Get timestepping coefficients
         get_coeff(times, lhs_coefs, rhs_coefs);

         // First, compute current timestep forcing -- get user forcing first

         /* In general, tracer and velocity forcing is coupled.  Velocity
            forcing can depend on tracers (e.g. density), and tracer forcing
            can depend on velocity and other tracers (reaction problems,
            perturbation density about a background state).  So, this means
            we need to construct a big honking forcing call. */

         /* First, set tracers_now and tracers_forcing_now to point to the
            proper (current) timelevels. */
         for (unsigned int i = 0; i < tracers.size(); i++) {
            tracers_forcing_now[i] = &((*tracers_forcing[i])[0]);
            tracers_now[i] = &((*tracers[i])[0]);
         }
         usercode->forcing(times[0], us[0], us_forcing[0],
                                    vs[0], vs_forcing[0],
                                    ws[0], ws_forcing[0],
                                    tracers_now, tracers_forcing_now);
         //fprintf(stderr,"%g %g %g\n",pvmax(us_forcing[0]),pvmax(vs_forcing[0]),pvmax(ws_forcing[0]));
         //fprintf(stderr,"%g %g %g\n",pvmax(us[0]),pvmax(vs[0]),pvmax(ws[0]));
//         usercode->forcing(times[0], us_forcing[0], vs_forcing[0],
//               ws_forcing[0], tracers_now);

         // Next, add advection. 

         // To prevent aliasing, we use the mixed form (tensor notation):
         //(u_i d/dx_i u_j + d/dx_i (u_i * u_j))/2 
         // U
         // FIXME: Is the sign on this advectin correct?
         // traditional advection gradient
         grad(us[0],*dx,*dy,*dz,ODD,EVEN,EVEN);
         //fprintf(stderr,"%g %g %g\n",pvmax(*dx),pvmax(*dy),pvmax(*dz));
         us_forcing[0] = us_forcing[0] - (us[0]*(*dx) + vs[0]*(*dy) + 
                        ws[0]*(*dz))/2;
         *dx = us[0]*us[0];
         *dy = us[0]*vs[0];
         *dz = us[0]*ws[0];
         divergence(*dx,*dy,*dz,*sca_temp,EVEN,ODD,ODD); // divergence for second half
         //fprintf(stderr,"%g\n",pvmax(*sca_temp));
         us_forcing[0] = us_forcing[0] - *sca_temp/2;
         // V
         grad(vs[0],*dx,*dy,*dz,EVEN,ODD,EVEN);
         //fprintf(stderr,"%g %g %g\n",pvmax(*dx),pvmax(*dy),pvmax(*dz));
         vs_forcing[0] = vs_forcing[0] - (us[0]*(*dx) + vs[0]*(*dy) +
                        ws[0]*(*dz))/2;
         //NOTE: Future possible optimization here, dx here is dy above
         *dx = vs[0]*us[0]; 
         *dy = vs[0]*vs[0];
         *dz = vs[0]*ws[0];
         divergence(*dx,*dy,*dz,*sca_temp,ODD,EVEN,ODD);
         //fprintf(stderr,"%g\n",pvmax(*sca_temp));
         vs_forcing[0] = vs_forcing[0] - *sca_temp/2;
         // W
         grad(ws[0],*dx,*dy,*dz,EVEN,EVEN,ODD);
         //fprintf(stderr,"%g %g %g\n",pvmax(*dx),pvmax(*dy),pvmax(*dz));
         ws_forcing[0] = ws_forcing[0] - (us[0]*(*dx) + vs[0]*(*dy) + 
                        ws[0]*(*dz))/2;
         *dx = ws[0]*us[0];
         *dy = ws[0]*vs[0];
         *dz = ws[0]*ws[0];
         divergence(*dx,*dy,*dz,*sca_temp,ODD,ODD,EVEN);
         //fprintf(stderr,"%g\n",pvmax(*sca_temp));
         ws_forcing[0] = ws_forcing[0] - *sca_temp/2;

//         fprintf(stderr,"%g %g %g\n",pvmax(us_forcing[0]),pvmax(vs_forcing[0]),pvmax(ws_forcing[0]));

         /* Advect tracers.  Nothing fancy here. */
         for (unsigned int i = 0; i < tracers.size(); i++) {
            /* Curioser and curioser; a tracer does not fit neatly into the
               realm of an even or odd expansion in the case of (trig) free-
               slip boundaries, at least when coupled into the momentum
               equations as active tracers.

               Ideally, the tracer should have Neumann boundary conditions
               along any free-sliip surface.  But if coupled into the equations
               as a body force (as in g*rho), then consistency requires an
               "odd" interpretation -- meaning a sine expansion and 
               undesirable Dirichlet BC's.

               Going with the "Fake it until we make it" principle, advection
               should be done consistently, and we can take a body force as
               some weird square wave.  The pressure will adopt the (inverse)
               square wave to compensate, and momentum will still be accurate.*/


            grad((*tracers[i])[0],*dx,*dy,*dz,
                  (t_xbc_neu[i] == 0 ? ODD : EVEN),
                  (t_ybc_neu[i] == 0 ? ODD : EVEN),
                  (t_zbc_neu[i] == 0 ? ODD : EVEN));
            // If there is forcing of the tracer at the boundary, we need to copy over
            // the boundary condition into a safe temporary array.  This isn't super-
            // desirable, but the alternative is to restrict every subsequent
            // operation to just the interior of the domain.  This is awkward for 
            // some (apply_timestep), and outright impossible for the filtering.
//            fprintf(stderr,"Before BC copy");
//            cerr << (*tracers_forcing[i])[0];
            if (tx == NO_SLIP && usercode->tracer_bc_forcing() && t_diffusivity[i] != 0) {
               if (local_lbounds(1) == 0) { // We have the left point
                  left_bc = (*tracers_forcing[i])[0](0,blitz::Range::all(),blitz::Range::all());
               }
               if (local_lbounds(1) + local_extents(1) == (szx - 1)) { // And the right point
                  right_bc = (*tracers_forcing[i])[0](szx-1,blitz::Range::all(),blitz::Range::all());
               }
            }
            if (tz == NO_SLIP && usercode->tracer_bc_forcing() && t_diffusivity[i] != 0) {
               // If we have top/bottom points, then those are present on every process
               top_bc = (*tracers_forcing[i])[0](blitz::Range::all(),blitz::Range::all(),0);
               bottom_bc = (*tracers_forcing[i])[0](blitz::Range::all(),blitz::Range::all(),szz-1);
            }

            /* Tracer advection is done here with just the u dot grad T form;
               this does introduce aliasing error (compared with the skew-symmetric
               form), but the velocity field here is not guaranteed to be exactly
               divergence-free near no-slip boundaries (projection error).  Using
               the skew-symmetric form will run the risk of introducing new max/min
               to the solution, which can then lead to follow-on physical effects. 

               And, it makes pictures look ugly. */
            (*tracers_forcing[i])[0] = (*tracers_forcing[i])[0] -
               (us[0]*(*dx) + vs[0]*(*dy) + ws[0]*(*dz));
            /*  // This code is included for historical reference on what to to
                // if using the skew-symmetric form
            *dx = us[0]*(*tracers[i])[0];
            *dy = vs[0]*(*tracers[i])[0];
            *dz = ws[0]*(*tracers[i])[0];
            divergence(*dx,*dy,*dz,*sca_temp,ODD,ODD,ODD);
            (*tracers_forcing[i])[0] = (*tracers_forcing[i])[0] -
               *sca_temp/2;
               */
            apply_timestep((*tracers[i])[1],*tracers[i],*tracers_forcing[i]);

            /* Filter the tracer to prevent blowups */

            // FIXME: In the event of BC-forcing, this breaks the entire notion of filtering
            // here at this step (where the BCs, of course, haven't actually been APPLIED.)
            // The boundary values of the tracer field are consequently going to be junk
            // (a mix of real values + BCs), which may trigger unacceptable filtering.

            // There are two possible ways to fix this.  The first is to mvoe the filtering after
            // the diffusive solve, but this may cause breakage for periodic/symmetric boundaries
            // that work fine in this role.  The second (and probably more sensible option) is
            // to build a fake boundary by extrapolating the previous values; this will create
            // a smooth field at least.
            
            filter3((*tracers[i])[1], *filter_spec,
                  (Sx == REAL) ? (t_xbc_neu[i] == 0 ? SINE : COSINE) : Sx,
                  (Sy == REAL) ? (t_ybc_neu[i] == 0 ? SINE : COSINE) : Sy,
                  (Sz == REAL) ? (t_zbc_neu[i] == 0 ? SINE : COSINE) : Sz,
                  f_cutoff,f_order,f_strength); 

            
            if (t_diffusivity[i] == 0) { // No diffusivity on this tracer
               /* Scale out the timestep, since a viscosity solver isn't
                  here to do it for us */
               (*tracers[i])[1] = (*tracers[i])[1] / lhs_coefs[1];
            } else {
               /* Solve the diffusivity part of the tracer equation,
                  like viscosity in the momentum equations below */
               visc_solver->change_m(lhs_coefs[1]/t_diffusivity[i]);
               (*tracers[i])[1] = (*tracers[i])[1]/(-t_diffusivity[i]);
               if (!usercode->tracer_bc_forcing()) { // If there is no BC forcing, then set BC entries to 0
                  if (tx == NO_SLIP) {
                     if ((*tracers[i])[1].lbound(firstDim) == 0) {
                        (*tracers[i])[1](0,blitz::Range::all(),blitz::Range::all()) = 0;
                     }
                     if ((*tracers[i])[1].ubound(firstDim) == szx-1) {
                        (*tracers[i])[1](szx-1,blitz::Range::all(),blitz::Range::all()) = 0;
                     }
                  }
                  if (tz == NO_SLIP) {
                     (*tracers[i])[1](blitz::Range::all(),blitz::Range::all(),
                                      0) = 0;
                     (*tracers[i])[1](blitz::Range::all(),blitz::Range::all(),
                                      szz-1) = 0;
                  }
               } else {
                  if (tx == NO_SLIP) {
                     if (local_lbounds(1) == 0) {
                        (*tracers[i])[1](0,blitz::Range::all(),blitz::Range::all()) = left_bc;
                     }
                     if (local_lbounds(1) + local_extents(1) == (szx - 1)) {
                        (*tracers[i])[1](szx-1,blitz::Range::all(),blitz::Range::all()) = right_bc;
                     }
                  }
                  if (tz == NO_SLIP) {
                     (*tracers[i])[1](blitz::Range::all(),blitz::Range::all(),0) = top_bc;
                     (*tracers[i])[1](blitz::Range::all(),blitz::Range::all(),szz-1) = bottom_bc;
                  }
               }

//               fprintf(stderr,"After BC copy\n");
//               cerr << (*tracers[i])[1];
//               if (master()) fprintf(stderr,"Diffusivity solve tracer %u\n",i);
               visc_solver->solve((*tracers[i])[1],(*tracers[i])[1],
                     (Sx == REAL) ? (t_xbc_neu[i] == 0 ? SINE : COSINE) : Sx,
                     (Sy == REAL) ? (t_ybc_neu[i] == 0 ? SINE : COSINE) : Sy,
                     (Sz == REAL) ? (t_zbc_neu[i] == 0 ? SINE : COSINE) : Sz,
                     t_zbc_dir[i],t_zbc_neu[i],
                     t_xbc_dir[i],t_xbc_neu[i],
                     t_ybc_dir[i],t_ybc_neu[i]);
//               cerr << (*tracers[i])[1];
//               exit(1);
            }
         }
            

         // Next, construct projected velocities and compute pressure
         
         /* For conservation of temporary arrays, use ?s_forcing[1] for
            projected velocities. */
         apply_timestep(us_forcing[1], us, us_forcing);
         apply_timestep(vs_forcing[1], vs, vs_forcing);
         apply_timestep(ws_forcing[1], ws, ws_forcing);

         /* Filter here, before projecting, to keep compressibility zero
            even when aliasing error is removed */
         filter3(us_forcing[1],*filter_spec,
               (Sx == REAL) ? SINE : Sx,
               (Sy == REAL) ? COSINE : Sy,
               (Sz == REAL) ? COSINE : Sz, f_cutoff,f_order,f_strength);
         filter3(vs_forcing[1],*filter_spec,
               (Sx == REAL) ? COSINE : Sx,
               (Sy == REAL) ? SINE : Sy,
               (Sz == REAL) ? COSINE : Sz, f_cutoff,f_order,f_strength);
         filter3(ws_forcing[1],*filter_spec,
               (Sx == REAL) ? COSINE : Sx,
               (Sy == REAL) ? COSINE : Sy,
               (Sz == REAL) ? SINE : Sz, f_cutoff,f_order,f_strength);
//         fprintf(stderr,"%g %g %g\n",pvmax(us_forcing[1]),pvmax(vs_forcing[1]),pvmax(ws_forcing[1]));
         divergence(us_forcing[1], vs_forcing[1], ws_forcing[1], *sca_temp,
               ODD, ODD, ODD);

//         fprintf(stderr,"Divergence taken\n");
//         cerr << *sca_temp;

//         fprintf(stderr,"%g\n",pvmax(*sca_temp));
         // Now, we have div(u*) = grad^2 pressure, so solve

         /* If using a chebyshev discretization, set the boundary conditions
            for pressure */
//         *sca_temp = 0;

         if (!mapped_problem) {
            // If the problem is unmapped, there are no cross terms
            // to worry about
            if (tx == NO_SLIP) {
               /* Left boundary -- positive u (going right) needs to be adjusted
                  by a negative pressure gradient at the boundary */
               if (sca_temp->lbound(firstDim) == 0) {
                  (*sca_temp)(0,blitz::Range::all(),blitz::Range::all()) =
                     -us_forcing[1](0,blitz::Range::all(),blitz::Range::all());
               }
               /* Right boundary -- positive u gets adjusted by a positive pressure
                  gradient */
               if (sca_temp->ubound(firstDim) == szx-1) {
                  (*sca_temp)(szx-1,blitz::Range::all(),blitz::Range::all()) =
                     us_forcing[1](szx-1,blitz::Range::all(),blitz::Range::all());
               }
            }

            if (tz == NO_SLIP) {
               /* After applying the pressure below, ws_forcing[1] will
                  contain the candidate vertical velocities less scaling
                  by timestepping coefficients. So, after applying - (*dz),
                  we want ws_forcing[1] at the top and bottom to be zero. */

               /* Bottom boundary --
                  Positive w (flow going up) needs to be adjusted by a
                  -negative- pressure derivative at the boundary, that
                  is the boundary sucks fluid out to compensate for
                  inflow */
               (*sca_temp)(blitz::Range::all(),blitz::Range::all(),0) =
                  -ws_forcing[1](blitz::Range::all(),blitz::Range::all(),0);

               /* Top boundary --
                  Positive w (flow going up) gets adjusted by a positive
                  pressure derivative, to push fluid back in */
               (*sca_temp)(blitz::Range::all(),blitz::Range::all(),szz-1) =
                  ws_forcing[1](blitz::Range::all(),blitz::Range::all(),szz-1);
            }
         } else {
            // With a mapped problem, we have to use the Jacobian computed at
            // initialization to determine what portion of the fluid is actually
            // flowing out the boundary.
            using blitz::Range;
            Range all = Range::all();
            if (tx == NO_SLIP) {
               // First up are the x-boundaries, since by our definition the corners
               // are considered part of the z-boundaries.

               // The tangent is in the direction of [x_beta,z_beta], so the normal is in
               // [z_beta,-x_beta] (normalized)
//               fprintf(stderr,"NSIntegrator:%d no slip mapped x\n",__LINE__);
               if (x_alpha->lbound(firstDim) == 0) {
                  (*sca_temp)(0,all,all) =
                     ((*z_beta)(0,all,all)*us_forcing[1](0,all,all) -
                      (*x_beta)(0,all,all)*ws_forcing[1](0,all,all)) /
                     sqrt(pow((*z_beta)(0,all,all),2)+pow((*x_beta)(0,all,all),2));
                  if (any((*z_beta)(0,all,all)*(*x_alpha)(0,all,all) -
                           (*x_beta)(0,all,all)*(*z_alpha)(0,all,all) < 0)) {
                     (*sca_temp)(0,all,all) *= -1;
                  }
               }
               if (x_alpha->ubound(firstDim) == szx-1) {
                  (*sca_temp)(szx-1,all,all) =
                     ((*z_beta)(szx-1,all,all)*us_forcing[1](szx-1,all,all) -
                      (*x_beta)(szx-1,all,all)*ws_forcing[1](szx-1,all,all)) /
                     sqrt(pow((*z_beta)(szx-1,all,all),2)+pow((*x_beta)(szx-1,all,all),2));
                  if (any((*z_beta)(szx-1,all,all)*(*x_alpha)(szx-1,all,all) -
                           (*x_beta)(szx-1,all,all)*(*z_alpha)(szx-1,all,all) > 0)) {
                     (*sca_temp)(szx-1,all,all) *= -1;
                  }
               }
            }

//            fprintf(stderr,"NSIntegrator:%d no slip mapped z\n",__LINE__);
            (*sca_temp)(all,all,0) =
               (-((*z_alpha)(all,all,0)*(us_forcing[1](all,all,0)))+
               ((*x_alpha)(all,all,0))*(ws_forcing[1])(all,all,0)) /
               (sqrt(pow((*z_alpha)(all,all,0),2)+pow((*x_alpha)(all,all,0),2)));

            // At k = 0, on the Chebyshev grid the numerical coordinate is +1,
            // meaning that the -outward normal- is positive in that (beta) coordinate.
            // The beta vector in physical space is [x_beta,z_beta].  If that dot
            // the normal vector is < 0, we need to flip signs to make sure we specify
            // an outward normal

            // Repeating a comment from gmres_2d, this should really be a parallel synchronized
            // check.  But on well-defined grids, this is either uniformly true or false along
            // the boundary (otherwise there's an orientation fail), and that kind of problem
            // will error elsewhere anyway.

            if (any(-(*z_alpha)(all,all,0)*(*x_beta)(all,all,0) +
                     (*x_alpha)(all,all,0)*(*z_beta)(all,all,0) < 0)) {
               (*sca_temp)(all,all,0) *= -1;
            }

            
            // Top boundary
            (*sca_temp)(all,all,szz-1) =
               (-((*z_alpha)(all,all,szz-1)*(us_forcing[1](all,all,szz-1)))+
               ((*x_alpha)(all,all,szz-1))*(ws_forcing[1])(all,all,szz-1)) /
               (sqrt(pow((*z_alpha)(all,all,szz-1),2)+pow((*x_alpha)(all,all,szz-1),2)));

            // At k = szz-1, on the Chebyshev grid the numerical coordinate is -1,
            // so the -outward normal- is in the negative beta direction.  Same test as
            // above, save that the sign is reversed.
            if (any(-(*z_alpha)(all,all,szz-1)*(*x_beta)(all,all,szz-1) +
                     (*x_alpha)(all,all,szz-1)*(*z_beta)(all,all,szz-1) > 0)) {
               (*sca_temp)(all,all,szz-1) *= -1;
            }


         }

               

         
//         if (master()) fprintf(stderr,"Pressure solve\n");
         pres_solver->change_m(0);
         pres_solver->solve(*sca_temp, pressure,
               (Sx == REAL) ? COSINE : Sx,
               (Sy == REAL) ? COSINE : Sy,
               (Sz == REAL) ? COSINE : Sz,0,1);




         // Apply pressure to projected velocities (make divergence free)

         grad(pressure, *dx, *dy, *dz,EVEN,EVEN,EVEN);


         us_forcing[1] = us_forcing[1] - (*dx);
         vs_forcing[1] = vs_forcing[1] - (*dy);
         ws_forcing[1] = ws_forcing[1] - (*dz);

//         cout << ws_forcing[1];
//         exit(1);
//         fprintf(stderr,"%g %g %g\n",pvmax(us_forcing[1]),pvmax(vs_forcing[1]),pvmax(ws_forcing[1]));


         if (visco > 0) { // We have viscosity
            // Solve viscosity equations

            visc_solver->change_m(lhs_coefs[1]/visco);
            us_forcing[1] = us_forcing[1]/(-visco);
            vs_forcing[1] = vs_forcing[1]/(-visco);
            ws_forcing[1] = ws_forcing[1]/(-visco);

            /* Set (and possibly get) proper boundary conditions for
               velocity here */
            if (tz == NO_SLIP) {
               us_forcing[1](blitz::Range::all(),blitz::Range::all(),0) = 0;
               us_forcing[1](blitz::Range::all(),blitz::Range::all(),szz-1) = 0;
               vs_forcing[1](blitz::Range::all(),blitz::Range::all(),0) = 0;
               vs_forcing[1](blitz::Range::all(),blitz::Range::all(),szz-1) = 0;
               ws_forcing[1](blitz::Range::all(),blitz::Range::all(),0) = 0;
               ws_forcing[1](blitz::Range::all(),blitz::Range::all(),szz-1) = 0;
            }
            if (tx == NO_SLIP && us_forcing[1].lbound(firstDim) == 0) {
               us_forcing[1](0,blitz::Range::all(),blitz::Range::all()) = 0;
               vs_forcing[1](0,blitz::Range::all(),blitz::Range::all()) = 0;
               ws_forcing[1](0,blitz::Range::all(),blitz::Range::all()) = 0;
            }
            if (tx == NO_SLIP && us_forcing[1].ubound(firstDim) == szx-1) {
               us_forcing[1](szx-1,blitz::Range::all(),blitz::Range::all()) = 0;
               vs_forcing[1](szx-1,blitz::Range::all(),blitz::Range::all()) = 0;
               ws_forcing[1](szx-1,blitz::Range::all(),blitz::Range::all()) = 0;
            }


//         if (master()) fprintf(stderr,"U-velocity solve\n");
            visc_solver->solve(us_forcing[1],us[1],
                  (Sx == REAL) ? SINE : Sx,
                  (Sy == REAL) ? COSINE : Sy,
                  (Sz == REAL) ? COSINE : Sz,1,0);
//         if (master()) fprintf(stderr,"V-velocity solve\n");
            visc_solver->solve(vs_forcing[1],vs[1],
                  (Sx == REAL) ? COSINE : Sx,
                  (Sy == REAL) ? SINE : Sy,
                  (Sz == REAL) ? COSINE : Sz,1,0);
//         if (master()) fprintf(stderr,"W-velocity solve\n");
            visc_solver->solve(ws_forcing[1],ws[1],
                  (Sx == REAL) ? COSINE : Sx,
                  (Sy == REAL) ? COSINE : Sy,
                  (Sz == REAL) ? SINE : Sz,1,0);
         } else { // Rescale, like a tracer.  Filtering was already done pre-pressure
            us[1] = us_forcing[1] / lhs_coefs[1];
            vs[1] = vs_forcing[1] / lhs_coefs[1];
            ws[1] = ws_forcing[1] / lhs_coefs[1];
            
         }
         // Free temporaries
         release_temp(sca_temp);
         release_temp(dz); release_temp(dy); release_temp(dx);
      }

   template <class C>
      DTArray * FluidEvolve<C>::get_temp() {
         /* Get a temporary array.  Previously allocated, if possible */
         if (unused_temps.size() > 0) {
            DTArray * mytemp = unused_temps.back();
            unused_temps.pop_back();
            used_temps.insert(mytemp);
            return mytemp;
         } else {
            DTArray * mytemp = alloc_array();
//            if (!master()) fprintf(stderr,"Allocating new temporary array\n");
            used_temps.insert(mytemp);
            return mytemp;
         }
      }
   template <class C>
      void FluidEvolve<C>::release_temp(DTArray * erase) {
         /* Return a temporary array to the unused list */
         set<DTArray *>::iterator loc = used_temps.find(erase);
         if (loc == used_temps.end()) {
            fprintf(stderr,"ERROR: Freeing an array that isn't a temporary\n");
            abort();
         } else {
            unused_temps.push_back(*loc);
            used_temps.erase(loc);
         }
         return;
      }

   template <class C>
      void FluidEvolve<C>::grad(DTArray & vec, DTArray & dx, DTArray & dy,
                                DTArray & dz, Sym sx, Sym sy, Sym sz) {
         /* grad -- computes the physical gradient of array vec, with the
            x, y, and z components stored in dx, dy, dz respectively. */
         /* To impement free-slip trig boundary conditions, the type of
            derivative has to change based on variable symmetry.  To allow
            for appropriate generality, symmetry is specified in the call
            [and only used if the dimension expansion is free-slip] */

         /* x */

         gradient->setup_array(&vec,get_trans_type(tx,sx),
               get_trans_type(ty,sy),get_trans_type(tz,sz));
         gradient->get_dx(&dx);
         gradient->get_dy(&dy);
         gradient->get_dz(&dz);
         return;

#if 0

         if (szx > 1) {
            if (tx == PERIODIC) {
               //fprintf(stderr,"-%g-\n",pvmax(vec));
               deriv_fft(vec, *trans_x, dx);
               /* Scale by length */
               //fprintf(stderr,"-%g-\n",pvmax(dx));
               dx = dx * (2*M_PI/Lx);
               //fprintf(stderr,"-%g-\n",pvmax(dx));
            } else if (ty == FREE_SLIP){
               if (sx == ODD) {
                  deriv_dst(vec, *trans_x, dx);
               } else {
                  assert (sx == EVEN);
                  deriv_dct(vec,*trans_x,dx);
               }
               dx = dx * (M_PI/Lx);
            } else {
               abort();
            }
         } else {
            dx = 0;
         }
         if (szy > 1) {
            if (ty == PERIODIC) {
               deriv_fft(vec, *trans_y, dy);
               dy = dy * (2*M_PI/Ly);
            } else if (ty == FREE_SLIP) {
               if (sy == ODD) {
                  deriv_dst(vec, *trans_y, dy);
               } else {
                  assert (sy == EVEN);
                  deriv_dct(vec,*trans_y,dy);
               }
               dy = dy * (M_PI/Ly);
            } else { // Chebyshev
               abort();
            }
         } else {
            dy = 0;
         }
         if (szz > 1) {
            if (tz == PERIODIC) {
               deriv_fft(vec, *trans_z, dz);
               dz = dz * (2*M_PI/Lz);
            } else if (tz == FREE_SLIP) {
               if (sz == ODD) {
                  deriv_dst(vec, *trans_z, dz);
               } else {
                  assert (sz == EVEN);
                  deriv_dct(vec,*trans_z,dz);
               }
               dz = dz * (M_PI/Lz);
            } else {
               assert (tz == NO_SLIP);
               deriv_cheb(vec,*trans_z,dz);
               dz = dz * (-2/Lz);
            }
         } else {
            dz = 0;
         }
         return;
#endif
      }

   template <class C>
      void FluidEvolve<C>::divergence(DTArray & up, DTArray & vp, DTArray & wp,
                                 DTArray & div,Sym sx, Sym sy, Sym sz)  {
         /* Calculate divergence of [up,vp,wp], store in div */
         /* The symmetry here might need to be rethought with variable
            Jacobians.  As currently written, sx applies to up, sy to vp, etc;
            but if dimensions couple then we might break some of these assumptions.
            (It's not guaranteed since Cheby dimensions don't use symmetry) */
         gradient->setup_array(&up,get_trans_type(tx,sx),
               get_trans_type(ty,sy),get_trans_type(tz,sz));
         // du/dx
         gradient->get_dx(&div,false);
         gradient->setup_array(&vp,get_trans_type(tx,sx),
               get_trans_type(ty,sy),get_trans_type(tz,sz));
         /* Accumulate with d/dy and d/dz*/
         // dv/dy
         gradient->get_dy(&div,true);
         gradient->setup_array(&wp,get_trans_type(tx,sx),
               get_trans_type(ty,sy),get_trans_type(tz,sz));
         // dw/dz
         gradient->get_dz(&div,true);
#if 0
         assert(gradient->constant_jac());
         DTArray * temp = get_temp();
         *temp = 0;
         if (szx > 1) {
            if (tx == PERIODIC) {
               deriv_fft(up,*trans_x,*temp);
               div = (*temp)*(2*M_PI/Lx);
            } else {
               assert (tx == FREE_SLIP);
               if (sx == ODD) 
                  deriv_dst(up,*trans_x,*temp);
               else {
                  assert (sx == EVEN);
                  deriv_dct(up,*trans_x,*temp);
               }
               div = (*temp)*(M_PI/Lx);
            }
         }
         if (szy > 1) {
            if (ty == PERIODIC) {
               deriv_fft(vp,*trans_y,*temp);
               div = div + (*temp)*(2*M_PI/Ly);
            } else {
               assert (ty == FREE_SLIP);
               if (sy == ODD) 
                  deriv_dst(vp,*trans_y,*temp);
               else {
                  assert (sy == EVEN);
                  deriv_dct(vp,*trans_y,*temp);
               }
               div = div + (*temp)*(M_PI/Ly);
            }
         }
         if (szz > 1) { 
            if (tz == PERIODIC) {
               deriv_fft(wp,*trans_z,*temp);
               div = div + (*temp)*(2*M_PI/Lz);
            } else if (tz == FREE_SLIP){
               if (sz == ODD) {
                  deriv_dst(wp,*trans_z,*temp);
               } else {
                  assert (sz == EVEN);
                  deriv_dct(wp,*trans_z,*temp);
               }
               div = div + (*temp)*(M_PI/Lz);
            } else {
               assert (tz == NO_SLIP);
               deriv_cheb(wp,*trans_z,*temp);
               div = div + (*temp)*(-2/Lz);
            }
         }
         release_temp(temp);
         return;
#endif
      }
         

   template <class C>
      void FluidEvolve<C>::apply_timestep(Array <double, 3> & ar,
                     Stepped<DTArray> & lhs, Stepped<DTArray> & rhs) {
         /* Apply the explicit portion of the calculated timestep coefficients
            to arrays lhs/rhs, storing the result in ar */
         ar = rhs_coefs[0]*rhs[0] + rhs_coefs[-1]*rhs[-1] + 
              rhs_coefs[-2]*rhs[-2] - (lhs_coefs[0]*lhs[0] +
                    lhs_coefs[-1]*lhs[-1] + lhs_coefs[-2]*lhs[-2]);
         return;
      }

   template <class C>
      double FluidEvolve<C>::get_cfl() {
         /* Using the CFL condition and a safety factor, calculate the
            maximum stable timestep */
/*         fprintf(stderr,"Top-lefts: %f \n %f \n %f \n",us[0](0,0,0),
               vs[0](0,0,0), ws[0](0,0,0));*/
/*         fprintf(stderr,"Maximum velocities %f, %f, %f",max(abs(us[0])),
              max(abs(vs[0])), max(abs(ws[0]))); */
         double timex;
         if (tx == NO_SLIP) {
            blitz::firstIndex ii; blitz::secondIndex jj; blitz::thirdIndex kk;
            double pin = M_PI/(szx-1);
            timex = (Lx/2)*min(abs((pin*sin(kk*pin)+pin*pin/2*cos(kk*pin))/(1e-8 + abs(us[0](ii,jj,kk)))));
         } else {
            timex = (Lx/szx) / (1e-6 + max(abs(us[0])));
         }
         double timey = (Ly/szy) / (1e-6 + max(abs(vs[0])));
         double timez;
         if (tz == NO_SLIP) {
            blitz::firstIndex ii; blitz::secondIndex jj; blitz::thirdIndex kk;
            /* The CFL condition is min(abs(dz/w)), but for a Chebyshev
               grid dz is not a constant.  So, use two terms of a Taylor
               Series to give one */
            double pin = M_PI/(szz-1);
            timez = (Lz/2)*min(abs((pin*sin(kk*pin)+pin*pin/2*cos(kk*pin))/(1e-8 + abs(ws[0](ii,jj,kk)))));
         } else {
            timez = (Lz/szz) / (1e-6 + max(abs(ws[0])));
         }
         /* Return the global minimum over all processors: use psmin */
/*         if (master()) {
            fprintf(stderr,"Getting ts of %f, %f, %f\n",tx,ty,tz);
         }*/
         return 0.25*psmin(fmin(fmin(timex,timey),timez));
      }

   template <class C>
      void FluidEvolve<C>::shift_arrays() {
         /* Shift all of the Stepped arrays by one */
         us.shift(); vs.shift(); ws.shift();
         us_forcing.shift(); vs_forcing.shift(); ws_forcing.shift();
         times.shift();

         for (unsigned int i = 0; i < tracers.size(); i++) {
            tracers[i]->shift();
            tracers_forcing[i]->shift();
         }
      }

}
//#else
//#error "NSIntegrator.cpp must be included from within NSIntegrator.hpp"
#else
#endif
