#include "../BaseCase.hpp"
#include "../NSIntegrator.hpp"
#include "../TArray.hpp"

using namespace NSIntegrator;

class minimal : public BaseCase {
   public:
      int size_cube() const { return 1; }
      double length_cube() const { return 1; }
      void init_vels(DTArray & u, DTArray & v, DTArray & w) {
         u = 0; v = 0; w = 0;
      }
};

int main() {
   minimal foo;
   FluidEvolve<minimal> fluidstuff(&foo);
   fluidstuff.initialize();
   fluidstuff.do_run(-1);
   return 0;
}
