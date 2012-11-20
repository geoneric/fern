#include <iostream>
#include <cstring>
#include <memory>
#include "Ranally/Configure.h"
#include "Ranally/Command/convert_command.h"
#include "Ranally/Command/describe_command.h"
#include "Ranally/Command/execute_command.h"
#include "Ranally/Command/import_command.h"


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
        "  execute             Execute script\n"
        "  convert             Convert script\n"
        "  describe            Describe script\n"
        "  import              Import data\n"
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
        std::unique_ptr<ranally::Command> command;

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
        else if(std::strcmp(argv[1], "import") == 0) {
            command.reset(new ranally::ImportCommand(argc - 1, argv + 1));
        }
        else {
            ranally::showGeneralHelp();
            status = EXIT_FAILURE;
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
