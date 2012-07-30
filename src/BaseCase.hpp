/* BaseCase -- basic skeletal framework for a functional NSIntegrator usercode.
   Eliminates some of the boilerplate. */
#ifndef BASECASE_HPP
#define BASECASE_HPP 1

#include <blitz/array.h>
#include "TArray.hpp"
#include "NSIntegrator.hpp"

using namespace TArrayn;
using namespace NSIntegrator;
using blitz::Array;
using std::vector;

class BaseCase {
   /* To reduce boilerplate, wrap some of the long functions, only calling
      them if actually used by usercode.  For example, a tracer-free code
      does not need the long-form forcing function -- nor does one with
      only passive tracers */

   public:
      /* Tracers */
      virtual int numActive() const; // Number of active tracers
      virtual int numPassive() const; // Number of passive tracers
      virtual int numtracers() const; // Number of tracers (total)

      /* Grid generataion */
      virtual int size_x() const; // Grid points in x
      virtual int size_y() const; // Grid points in y
      virtual int size_z() const; // Grid points in z
      virtual int size_cube() const { abort(); }; // Special case -- N*N*N

      virtual double length_x() const; // Length in x
      virtual double length_y() const; // Length in y
      virtual double length_z() const; // Length in z
      virtual double length_cube() const {abort();}; // Special case -- L*L*L

      virtual DIMTYPE type_x() const; // Expansion type in x
      virtual DIMTYPE type_y() const; // Expansion type in y
      virtual DIMTYPE type_z() const; // Expansion type in z
      virtual DIMTYPE type_default() const; // Default expansion type

      /* Functions to provide for custom tracer boundaries, along any
         dimension whose expansion type supports them (generally that
         would be chebyshev-type).  The default tracer-BC behaviour
         will be to give neumann-type BCs. Dirichlet-type BCs are possible
         with a sine-based expansion rather than cosine-based expansion
         and have been hacked in with a global variable, but this mechanism
         is far more general.  One will expect an error/abort if an improper
         BC-type is requested for a boundary that doesn't actually support it
         (namely a Robin-type BC on a non-Chebyshev dimension */


      // Further note -- we want the SAME boundary condition at the minimum and
      // maximum of the domain.
      virtual void tracer_bc_x(int tracernum, double & dirichlet, double & neumann) const;
      virtual void tracer_bc_y(int tracernum, double & dirichlet, double & neumann) const;
      virtual void tracer_bc_z(int tracernum, double & dirichlet, double & neumann) const;

      // Whether ANY tracers will have nonzero boundary conditions.  If this is true
      // then the user forcing code is responsible for calculating/applying the proper
      // RHS at the boundaries.  If this is false (default), then the prior behaviour
      // of the integrator code zeroing BCs after forcing is retained.
      virtual bool tracer_bc_forcing() const;

      virtual bool is_mapped() const; // Whether this problem has mapped coordinates
      // Coordinate mapping proper, if is_mapped() returns true.  This features full,
      // 3D arrays, but at least initially we're restricting ourselves to 2D (x,z)
      // mappings
      virtual void do_mapping(DTArray & xgrid, DTArray & ygrid, DTArray & zgrid);

      /* Physical parameters */
      virtual double get_visco() const; // Physical viscosity
      virtual double get_diffusivity(int tracernum) const; // Diffusivity

      /* Initialization */
      virtual double init_time() const; // Initialization time
      virtual void init_tracers(vector<DTArray *> & tracers);
      virtual void init_vels(DTArray & u, DTArray & v, DTArray & w) { abort();};

      virtual void init_tracer(int t_num, DTArray & tracer) { abort();}; // single-tracer

      /* Numerical checks */
      virtual double check_timestep(double step, double now);

      // Get incoming gradient operator, for differentials in analysis.  This is a null-
      // op unless the user code cares, in which case it will override this to store
      // the gradient.
      virtual void set_grad(Grad * in_grad) {return;};

      /* Forcing */
      /* Big, ultra-general forcing function */
      virtual void forcing(double t, DTArray & u, DTArray & u_f,
                                 DTArray & v, DTArray & v_f,
                                 DTArray & w, DTArray & w_f,
                                 vector<DTArray *> & tracers,
                                 vector<DTArray *> & tracers_f);
      /* No-active-tracers specialization */
      virtual void passive_forcing(double t, DTArray & u, DTArray & u_f,
                                 DTArray & v, DTArray & v_f,
                                 DTArray & w, DTArray & w_f);
      /* Independent-of-current-velocity no-tracer forcing */
      virtual void stationary_forcing(double t, DTArray& u_f, DTArray& v_f, 
            DTArray& w_f);
      
      /* If there are active tracers, split V and T focing */
      virtual void vel_forcing(double t, DTArray& u_f, DTArray& v_f,
                           DTArray& w_f, vector<DTArray *> & tracers) {abort();};
      virtual void tracer_forcing(double t, DTArray & u,
               DTArray & v, DTArray & w,
               vector<DTArray *> & tracers_f) {abort();};

      /* Analysis and writing */

      virtual void analysis(double t, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer, DTArray & pres); // General Analysis
      virtual void analysis(double t, DTArray & u, DTArray & v, DTArray & w,
            vector<DTArray *> tracer); // Less pressure
      virtual void vel_analysis(double t, DTArray & u, DTArray & v, 
            DTArray & w) {abort();}; // Velocity analysis
      virtual void tracer_analysis(double t, int t_num, DTArray & tracer) {abort();};
         // Single-tracer analysis
};

extern template class FluidEvolve<BaseCase>;
typedef FluidEvolve<BaseCase> EasyFlow; // Explicit template instantiation
#endif
