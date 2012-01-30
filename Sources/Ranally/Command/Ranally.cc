#include <iostream>
#include <cstring>
#include <boost/scoped_ptr.hpp>
#include "Ranally/Configure.h"
#include "ConvertCommand.h"
#include "DescribeCommand.h"
#include "ExecuteCommand.h"



namespace ranally {

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
    "  execute             "
      "Execute script (this is the default when no command is\n"
    "                      provided)\n"
    "  convert             Convert script\n"
    "  describe            Describe script\n"
    "\n"
    "See 'ranally COMMAND --help' for more information on a specific command.\n"
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

} // namespace ranally



int main(
  int argc,
  char** argv)
{
  int status = EXIT_FAILURE;

  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    ranally::showGeneralHelp();
    status = EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--version") == 0) {
    ranally::showVersion();
    status = EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--build") == 0) {
    ranally::showBuild();
    status = EXIT_SUCCESS;
  }
  else {
    boost::scoped_ptr<ranally::Command> command;

    // A command may be given. Find out which one.
    if(std::strcmp(argv[1], "convert") == 0) {
      command.reset(new ranally::ConvertCommand(argc - 1, argv + 1));
    }
    else if(std::strcmp(argv[1], "describe") == 0) {
      command.reset(new ranally::DescribeCommand(argc - 1, argv + 1));
    }
    else if(std::strcmp(argv[1], "execute") == 0) {
      command.reset(new ranally::ExecuteCommand(argc - 1, argv + 1));
    }
    else {
      // Default command is 'execute'.
      command.reset(new ranally::ExecuteCommand(argc, argv));
    }

    assert(command);

    try {
      status = command->execute();
    }
    catch(std::exception const& exception) {
      std::cerr << exception.what() << '\n';
      status = EXIT_FAILURE;
    }
  }

  return status;
}

