#include <cstring>
#include <iostream>
#include <memory>
#include "ranally/configure.h"
#include "ranally/command/convert_command.h"
#include "ranally/command/describe_command.h"
#include "ranally/command/execute_command.h"
#include "ranally/command/import_command.h"
#include "ranally/command/interpreter.h"
#include "ranally/command/message.h"


namespace ranally {

void show_general_help()
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
        "The interactive interpreter is entered when no arguments are passed.\n"
        ;
}


void show_build()
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

    if(argc == 1) {
        // No arguments, enter the interpreter.
        try {
            ranally::enter_interpreter();
            status = EXIT_SUCCESS;
        }
        catch(std::exception const& exception) {
            std::cerr << ranally::String(exception.what()) << '\n';
            status = EXIT_FAILURE;
        }
    }
    else if(std::strcmp(argv[1], "--help") == 0) {
        // The help option.
        ranally::show_general_help();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--version") == 0) {
        ranally::show_version();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--build") == 0) {
        ranally::show_build();
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
            ranally::show_general_help();
            status = EXIT_FAILURE;
        }

        assert(command);

        try {
            status = command->execute();
        }
        catch(std::exception const& exception) {
            std::cerr << ranally::String(exception.what()) << '\n';
            status = EXIT_FAILURE;
        }
    }

    return status;
}
