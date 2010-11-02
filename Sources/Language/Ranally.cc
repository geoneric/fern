#include <iostream>
#include <boost/program_options.hpp>

#include "Configure.h"



namespace po = boost::program_options;

int main(
  int argc,
  char** argv)
{
  po::options_description optionsDescription("Allowed options");
  optionsDescription.add_options()
    ("help", "show help message")
    ("build", "show build info")
    ("version", "show version")
    ;

  // TODO Think of all tasks this utility needs to perform and design a
  //      command line interface for that.
  //      - interpret/execute script
  //      - convert script to script (roundtrip)
  //      - convert script to dot-graph
  //      - convert script to C++
  //      - convert script to C++, including Python extension
  // Some of these actions can happen simultaneously. Others are silly to
  // combine.
  //   valid: all conversions
  //   silly: interpret and convert

  // Ideas:
  // - Execute is default/implied if no conversion action is provided.
  // - Convert argument has a value of script | dot | C++ | Python
  // - Iff convert argument is provided, than no execution takes place.
  //   Positional arguments are related to the conversion process
  //   (target dir / file, ...).

  po::variables_map variablesMap;
  po::store(po::parse_command_line(argc, argv, optionsDescription),
    variablesMap);
  po::notify(variablesMap);

  if(variablesMap.empty() || variablesMap.count("help")) {
    std::cout << optionsDescription << "\n";
    return EXIT_SUCCESS;
  }

  if(variablesMap.count("version")) {
    std::cout << "ranally " << RANALLY_VERSION << "-" << RANALLY_BUILD_STAGE
      << "\n";
    std::cout << RANALLY_COPYRIGHT << "\n";
    return EXIT_SUCCESS;
  }

  if(variablesMap.count("build")) {
    std::cout << RANALLY_BUILD_TYPE << " build (" << __DATE__ << ")\n"
      << RANALLY_ARCHITECTURE << ", "
      << RANALLY_SYSTEM << ", "
      << RANALLY_CXX_COMPILER << ", "
      << "\n";
    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}
