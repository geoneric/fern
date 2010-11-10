#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
// #include <boost/program_options.hpp>

#include "dev_UnicodeUtils.h"

#include "AlgebraParser.h"
#include "Configure.h"
#include "DotVisitor.h"
#include "XmlParser.h"
#include "ScriptVertex.h"
#include "ScriptVisitor.h"



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
//
// - execute command is implied.
// - first positional is filename of script.
// - if --script option is provided, than script positional should not be.
//   ranally model.ran
//   ranally execute model.ran
//   ranally --script "slope = slope("dem")"
//   ranally execute --script "slope = slope("dem")"
//
// - first positional is language argument.
// - 
//   ranally convert dot model.ran model.dot
//   ranally convert c++ model.ran
//   ranally convert c++ --namespace bla --output-directory ./blo model.ran
//   ranally convert python --package bli model.ran
//
//   The Python extension converter should not create the core C++ code.
//   The c++ converter is for doing that. Or can we use two convert commands?



// namespace po = boost::program_options;
//

namespace {

void showGeneralHelp()
{
  std::cout <<
    "usage: ranally [--help] [--build] [--version] [COMMAND] [ARGS]\n"
    "\n"
    "--help                Show help message\n"
    "--version             Show version\n"
    "--build               Show build info\n"
    "\n"
    "commands:\n"
    "  execute             Execute script (this is the default when no\n"
    "                      command is provided)\n"
    "  convert             Convert script\n"
    "\n"
    "See 'ranally COMMAND --help' for more information on a specific command.\n"
    ;
}



void showConvertHelp()
{
  std::cout <<
    "usage: ranally convert LANGUAGE SCRIPT\n"
    "\n"
    "Convert the script to a target language.\n"
    "\n"
    "languages:\n"
    "  ranally             Round-trip script\n"
    "  dot                 Convert script to Dot graph\n"
    "  c++                 Convert script to C++ code\n"
    "  python              Convert script to C++ code for Python extension\n"
    "\n"
    "See 'ranally convert LANGUAGE --help' for more information on a specific\n"
    "language.\n"
    ;
}



void showConvertDotHelp()
{
  std::cout <<
    "usage: ranally convert dot [input script] [output script]\n"
    "\n"
    "Convert the script to a dot graph containing the syntax tree.\n"
    "\n"
    "  input script        Script to convert (standard input is read if no\n"
    "                      script arguments are provided)\n"
    "  output script       File to write result to (standard output is\n"
    "                      written if this argument is not provided)\n"
    ;
}



void showConvertRanallyHelp()
{
  std::cout <<
    "usage: ranally convert ranally [input script] [output script]\n"
    "\n"
    "Convert the script to a ranally script (round-trip).\n"
    "\n"
    "  input script        Script to convert (standard input is read if no\n"
    "                      script arguments are provided)\n"
    "  output script       File to write result to (standard output is\n"
    "                      written if this argument is not provided)\n"
    ;
}



void showExecuteHelp()
{
  std::cout <<
    "usage: ranally [execute] SCRIPT\n"
    "\n"
    "Execute the script. The execute command is implied if not provided.\n"
    ;
}



void showVersion()
{
  std::cout << "ranally " << RANALLY_VERSION << "-" << RANALLY_BUILD_STAGE
    << "\n";
  std::cout << RANALLY_COPYRIGHT << "\n";
}



void showBuild()
{
  std::cout << RANALLY_BUILD_TYPE << " build (" << __DATE__ << ")\n"
    << RANALLY_ARCHITECTURE << ", "
    << RANALLY_SYSTEM << ", "
    << RANALLY_CXX_COMPILER
    << "\n";
}

} // Anonymous namespace



int main(
  int argc,
  char** argv)
{
  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    showGeneralHelp();
    return EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--version") == 0) {
    showVersion();
    return EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--build") == 0) {
    showBuild();
    return EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "convert") == 0) {
    if(argc == 2 || std::strcmp(argv[2], "--help") == 0) {
      // No arguments, or the help option.
      showConvertHelp();
      return EXIT_SUCCESS;
    }

    assert(argc >= 3);

    if(std::strcmp(argv[2], "dot") == 0) {
      if(argc == 3 || std::strcmp(argv[3], "--help") == 0) {
        showConvertDotHelp();
        return EXIT_SUCCESS;
      }
      else if(argc == 4) {
        showConvertDotHelp();
        return EXIT_FAILURE;
      }

      assert(argc == 5);

      std::cout << "Convert (dot) " << argv[3] << " to " << argv[4] << "\n";

      return EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[2], "c++") == 0) {
      std::cout << "Convert to c++...\n";
      return EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[2], "python") == 0) {
      std::cout << "Convert to python...\n";
      return EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[2], "ranally") == 0) {
      if(argc >= 4 && std::strcmp(argv[3], "--help") == 0) {
        showConvertRanallyHelp();
        return EXIT_SUCCESS;
      }

      ranally::ScriptVisitor visitor;
      ranally::AlgebraParser parser;
      UnicodeString xml;

      if(argc == 3) {
        // Read script from the standard input stream.
        std::ostringstream script;
        script << std::cin.rdbuf();
        xml = ranally::AlgebraParser().parseString(UnicodeString(
          script.str().c_str()));
      }
      else if(argc == 4) {
        // Read script from a file.
        std::string inputFileName(argv[3]);
        xml = ranally::AlgebraParser().parseFile(UnicodeString(
          inputFileName.c_str()));
      }

      UnicodeString script = ranally::XmlParser().parse(xml)->Accept(visitor);

      if(argc <= 4) {
        std::cout << dev::encodeInUTF8(script) << std::endl;
      }
      else if(argc == 5) {
        std::cout << dev::encodeInUTF8(script) << std::endl;
        std::cout << "TODO write to file\n";
      }

      return EXIT_SUCCESS;
    }
    else {
      std::cerr << "Unknown target language...\n";
      std::cerr << "Conversion help...\n";
      return EXIT_FAILURE;
    }
  }
  else if(std::strcmp(argv[1], "execute") == 0) {
    if(argc == 2 || std::strcmp(argv[2], "--help") == 0) {
      // No arguments, or the help option.
      showExecuteHelp();
      return EXIT_SUCCESS;
    }

    std::cout << "Execute script...\n";
    return EXIT_SUCCESS;
  }
  else {
    std::cout << "Execute script...\n";
    return EXIT_SUCCESS;
  }





  // po::options_description optionsDescription("Allowed options");
  // optionsDescription.add_options()
  //   ("help", "show help message")
  //   ("build", "show build info")
  //   ("version", "show version")
  //   ;

  // po::variables_map variablesMap;
  // po::store(po::parse_command_line(argc, argv, optionsDescription),
  //   variablesMap);
  // po::notify(variablesMap);

  // if(variablesMap.empty() || variablesMap.count("help")) {
  //   std::cout << optionsDescription << "\n";
  //   return EXIT_SUCCESS;
  // }

  // if(variablesMap.count("version")) {
  //   std::cout << "ranally " << RANALLY_VERSION << "-" << RANALLY_BUILD_STAGE
  //     << "\n";
  //   std::cout << RANALLY_COPYRIGHT << "\n";
  //   return EXIT_SUCCESS;
  // }

  // if(variablesMap.count("build")) {
  //   std::cout << RANALLY_BUILD_TYPE << " build (" << __DATE__ << ")\n"
  //     << RANALLY_ARCHITECTURE << ", "
  //     << RANALLY_SYSTEM << ", "
  //     << RANALLY_CXX_COMPILER
  //     << "\n";
  //   return EXIT_SUCCESS;
  // }

  return EXIT_SUCCESS;
}
