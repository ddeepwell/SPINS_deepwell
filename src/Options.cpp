// Command line and configuration-file options parser for SPINS,
// with a simplified interface for user-code

/* This options parser uses the blitz program_options library to do
   the dirty work; this (reasonably easily) allows the use of the
   command line to do things like:

   ./run_case --Nx=100 --Ny=200 --Nz=100 --until=10.0

   or

   ./run_case --restart --sequence=10 --time=1.0 --output=0.01

   depending on what the case itself needs.  Additionally, this
   library also easily allows for the support of a key=value - style
   configuration file, which is ABSOLUTELY NEEDED for properly-
   documented user cases.  As of this writing, the configuration
   "style" has been to have an ordered-but-unlabeled sequence of values
   in a hard-coded configuration file.  This is fine as an alternative
   to recompilation for each case, but it is damn-near illegible weeks
   later.
*/

#include <iostream> // Streaming IO
#include <fstream> // File streams
#include <vector> // Vector
#include <mpi.h>
#include <Par_util.hpp>
#include <Options.hpp>

// Include boost program options library
#include <boost/program_options.hpp> 

// Use the boost program options namespace
namespace popt = boost::program_options;

// Use std namespace for vector and I/O
using namespace std;

// Global vector for separate options categories; this will still all
// be parsed as a giant options list at run-time, but separating 
// options into categories will provide some nice display bonuses.
vector<popt::options_description *> categories;

// Create a new option category.  This does not check to see if
// a category has already been created, so repeating an already-
// used category description may create confusing results
void option_category(const char * name) {
   categories.push_back(new popt::options_description(name));
}

// Wrapper for adding a "switch" option that is either present (true) or
// not present (false).
void add_switch(const char * name, bool * location, const char * description) {
   categories.back()->add_options()
      (name,popt::bool_switch(location),description);
}

// Global variables for help and config file
string config_file;
bool help;

// Function to initialize the options-categories, and include
// highly-necessary baseline options (help and config file)
void options_init() {
   option_category("SPINS: baseline options");
   add_option("config",&config_file,string("spins.conf"),"Configuration file");
   add_switch("help",&help,"Print this set of options and exit");
}

// Run the options parser, using the passed-in argc/argv from
// main().  This also opens and reads in the configuration file.
void options_parse(int argc, char ** argv) {

   // First, combine all of the categorized options into a
   // full list

   popt::options_description combined_options;

   // Iterate over the categories
   for (vector<popt::options_description *>::iterator
         opts = categories.begin();
         opts != categories.end();
         ++opts) {
      // And add it to the combined list
      combined_options.add(**opts);
   }

   // Holder for variables received
   popt::variables_map vm;

   popt::parsed_options parsed(&combined_options);
   try { 
      // Parse the command line
      parsed = popt::command_line_parser(argc,argv).
            options(combined_options).
            allow_unregistered().
            run();
         popt::store(parsed,vm);
      notify(vm);
   } catch ( exception& e) {
      // Program options throws an exception if there
      // are any invalid options
      if (master()) {
         cerr << "Command line parsing error:" << endl;
         cerr << "   " << e.what() << endl;
      }
      help = true;
   }

   // Collect any unknown options, so that we can warn about them
   std::vector<string> unknown_options =
      popt::collect_unrecognized(parsed.options,popt::include_positional);

   if (master() && unknown_options.size() > 0) {
      cerr << 
         "WARNING: Unknown options were passed on the command line" <<
         endl << "They were:" << endl;

      for (vector<string>::iterator opt = unknown_options.begin();
            opt != unknown_options.end();
            ++opt) {
         cerr << "   " << *opt << endl;
      }
   }


   // Now, parse the configuration file
   std::ifstream config_stream(config_file.c_str());
   if (!help && !config_stream) {
      if (master()) {
         cerr << "ERROR: Configuration file " << config_file <<
            " could not be opened\n";
      }
      help = true;
   } else if (!help) {
      try {
         // Gather the options from the configuration file stream
         parsed = popt::parse_config_file(
               config_stream,
               combined_options,
               true);
         popt::store(parsed,vm);
      } catch (exception &e ) {
         // catch invalid option-values
         if (master()) {
            cerr << "Configuration file parsing error:" << endl;
            cerr << "   " << e.what() << endl;
         }
         help = true;
      }
      notify(vm);
      

      /* NOTE: THIS ENTIRE SEQUENCE IS BUGGED IN BOOST VERSION 1.40
         (see svn.boost.org/trac/boost/ticket/2727 ).

         This is fixed in Boost 1.42 and later.*/
      // Collect any unrecognized options
      unknown_options = popt::collect_unrecognized(parsed.options,popt::include_positional);

      // ... and warn for them
      if (master() && unknown_options.size() > 0) {
         cerr << "WARNING: Unrecognized options were present in the configuration file" <<
            endl;
         cerr << "They were:" << endl;
         for (vector<string>::iterator opt = unknown_options.begin();
               opt != unknown_options.end();
               ++opt) {
            cerr << "   " << *opt << endl;
         }
      }
   }
   
   if (help) {
      if (master()) {
         cerr << combined_options << endl;
      }
      MPI_Finalize(); exit(1);
   }

}

// Add_option specialization so string default values are nicely
// handled as "constant character arrays in double quotes"
void add_option(
      const char * name, string * location,
      const char * def_value, const char * description) {
   add_option<string>(name, location, string(def_value), description);
}

