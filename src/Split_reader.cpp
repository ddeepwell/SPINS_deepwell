#include "Split_reader.hpp"
template blitz::Array<double,2> * read_2d_slice<double> (
      const char *, int, int, blitz::Range, blitz::Range,
      blitz::GeneralArrayStorage<2> );

template blitz::Array<float,2> * read_2d_slice<float> (
      const char *, int, int, blitz::Range, blitz::Range,
      blitz::GeneralArrayStorage<2> );
