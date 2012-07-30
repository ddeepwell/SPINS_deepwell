#include "BaseCase.hpp"
#include "NSIntegrator.hpp"
#include "TArray.hpp"

//using namespace TArray;
using namespace NSIntegrator;
using blitz::Array;
using std::vector;

/* Implementation of non-abstract methods in BaseCase */

int BaseCase::numActive() const { return 0; }
int BaseCase::numPassive() const { return 0; }
int BaseCase::numtracers() const { /* total number of tracers */
   return numActive() + numPassive();
}

int BaseCase::size_x() const {
   return size_cube();
}
int BaseCase::size_y() const {
   return size_cube();
}
int BaseCase::size_z() const {
   return size_cube();
}
double BaseCase::length_x() const {
   return length_cube();
}
double BaseCase::length_y() const {
   return length_cube();
}
double BaseCase::length_z() const {
   return length_cube();
}

DIMTYPE BaseCase::type_x() const {
   return type_default();
}
DIMTYPE BaseCase::type_y() const {
   return type_default();
}
DIMTYPE BaseCase::type_z() const {
   return type_default();
}
DIMTYPE BaseCase::type_default() const {
   return PERIODIC;
}

void BaseCase::tracer_bc_x(int t_num, double & dir, double & neu) const {
   if (!zero_tracer_boundary) {
      dir = 0; 
      neu = 1;
   }
   else {
      dir = 1;
      neu = 0;
   }
   return;
}
void BaseCase::tracer_bc_y(int t_num, double & dir, double & neu) const {
   if (!zero_tracer_boundary) {
      dir = 0; 
      neu = 1;
   }
   else {
      dir = 1;
      neu = 0;
   }
   return;
}
void BaseCase::tracer_bc_z(int t_num, double & dir, double & neu) const {
   if (!zero_tracer_boundary) {
      dir = 0; 
      neu = 1;
   }
   else {
      dir = 1;
      neu = 0;
   }
   return;
}
bool BaseCase::tracer_bc_forcing() const {
   return false;
}
bool BaseCase::is_mapped() const { // Whether this problem has mapped coordinates
   return false;
}
// Coordinate mapping proper, if is_mapped() returns true.  This features full,
// 3D arrays, but at least initially we're restricting ourselves to 2D (x,z)
// mappings
void BaseCase::do_mapping(DTArray & xgrid, DTArray & ygrid, DTArray & zgrid) {
   return;
}

/* Physical parameters */
double BaseCase::get_visco() const {
   return 0;
}
double BaseCase::get_diffusivity(int t) const {
   return 0;
}


/* Initialization */
double BaseCase::init_time() const {
   return 0;
}
void BaseCase::init_tracers(vector<DTArray *> & tracers) {
   /* Initalize tracers one-by-one */
   if (numtracers() == 0) return; // No tracers, do nothing
   assert(numtracers() == tracers.size()); // Sanity check
   for (int i = 0; i < numtracers(); i++) {
      init_tracer(i, *(tracers[i]));
   }
}

double BaseCase::check_timestep(double step, double now) {
   return step; // Default, accept the given stepsize
}

/* Forcing */
void BaseCase::forcing(double t,  DTArray & u, DTArray & u_f,
       DTArray & v, DTArray & v_f,  DTArray & w, DTArray & w_f,
      vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
   /* First, if no active tracers then use the simpler form */
   if (numActive() == 0) {
      passive_forcing(t, u, u_f, v, v_f, w, w_f);
   } else {
      /* Look at split velocity-tracer forcing */
      vel_forcing(t, u_f, v_f, w_f, tracers);
      tracer_forcing(t, u, v, w, tracers_f);
   }
   /* And any/all passive tracers get 0 forcing */
   for (int i = 0; i < numPassive(); i++) {
      *(tracers_f[numActive() + i]) = 0;
   }
}
void BaseCase::passive_forcing(double t,  DTArray & u, DTArray & u_f,
       DTArray & v, DTArray & v_f,  DTArray &, DTArray & w_f) {
   /* Reduce to velocity-independent case */
   stationary_forcing(t, u_f, v_f, w_f);
}
void BaseCase::stationary_forcing(double t, DTArray & u_f, DTArray & v_f, 
      DTArray & w_f) {
   /* Default case, 0 forcing */
   u_f = 0;
   v_f = 0;
   w_f = 0;
}

/* Analysis */
void BaseCase::analysis(double t, DTArray & u, DTArray & v, DTArray & w,
      vector<DTArray *> tracer, DTArray & pres) {
   analysis(t,u,v,w,tracer);
}
void BaseCase::analysis(double t, DTArray & u, DTArray & v, DTArray & w,
      vector<DTArray *> tracer) {
   /* Do velocity and tracer analysis seperately */
   vel_analysis(t, u, v, w);
   for (int i = 0; i < numtracers(); i++) {
     tracer_analysis(t, i, *(tracer[i]));
   } 
}

