// Command line and configuration-file options parser for SPINS,
// with a simplified interface for user-code

#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include <vector> // Vector

// Include boost program options library
#include <boost/program_options.hpp> 

// Use the boost program options namespace
namespace popt = boost::program_options;

// Use std namespace for vector and I/O
using namespace std;

// Global vector for separate options categories; this will still all
// be parsed as a giant options list at run-time, but separating 
// options into categories will provide some nice display bonuses.
extern vector<popt::options_description *> categories;

// Create a new option category.  This does not check to see if
// a category has already been created, so repeating an already-
// used category description may create confusing results
void option_category(const char * name);

// Wrapper for --key=value (or --key value, or key=value in config-
// file) option, pushed into the current category
template <class t> void add_option(const char * name, t * location, const char * description) {
   categories.back()->add_options()
      (name,popt::value<t>(location),description);
}

// Type-agnostic wrapper for adding a key=value option that has a
// default value
template <class t> void add_option(
      const char * name, t * location, const t & def_value, const char * description) {
   categories.back()->add_options()
      (name,popt::value<t>(location)->default_value(def_value),description);
}

// Wrapper for adding a "switch" option that is either present (true) or
// not present (false).
void add_switch(const char * name, bool * location, const char * description);

// Function to initialize the options-categories, and include
// highly-necessary baseline options (help and config file)
void options_init();

// Run the options parser, using the passed-in argc/argv from
// main().  This also opens and reads in the configuration file.
void options_parse(int argc, char ** argv);
#endif
