#include <iostream>
#include <boost/program_options.hpp>



namespace po = boost::program_options;

int main(
  int argc,
  char** argv)
{
  po::options_description optionsDescription("Allowed options");
  optionsDescription.add_options()
    ("help", "produce help message")
    ("version", "show version")
    ;

  po::variables_map variablesMap;
  po::store(po::parse_command_line(argc, argv, optionsDescription),
    variablesMap);
  po::notify(variablesMap);

  if(variablesMap.count("help")) {
    std::cout << optionsDescription << "\n";
    return EXIT_SUCCESS;
  }

  // TODO Make a header that is configured at configure time and contains
  //      some platform information from CMake variables.
  //      Check uname -a output.
  if(variablesMap.count("version")) {
    std::cout << "ranally <version>\n";
    std::cout << "Build on <date> for <os> on <architecture>\n";
    std::cout << "Copyright (C) 2010 Kor de Jong\n";
    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}
