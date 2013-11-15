#include <cstring>
#include <iostream>
#include <memory>
#include "fern/configure.h"
#include "fern/command/compile_command.h"
#include "fern/command/describe_command.h"
#include "fern/command/execute_command.h"
#include "fern/command/import_command.h"
#include "fern/command/interpreter.h"
#include "fern/command/message.h"


namespace fern {

void show_general_help()
{
    std::cout <<
        "usage: fern [--help] [--build] [--version] [COMMAND] [ARGS]\n"
        "\n"
        "--help                Show help message\n"
        "--version             Show version\n"
        "--build               Show build info\n"
        "\n"
        "commands:\n"
        "  execute             Execute script\n"
        "  compile             Compile script\n"
        "  describe            Describe script\n"
        "  import              Import data\n"
        "\n"
        "See 'fern COMMAND --help' for more information on a specific command.\n"
        "The interactive interpreter is entered when no arguments are passed.\n"
        ;
}


void show_build()
{
    std::cout << FERN_BUILD_TYPE << " build (" << __DATE__ << ")\n"
        << FERN_ARCHITECTURE << ", "
        << FERN_SYSTEM << ", "
        << FERN_CXX_COMPILER
        << "\n";
}

} // namespace fern


int main(
    int argc,
    char** argv)
{
    int status = EXIT_FAILURE;

    if(argc == 1) {
        // No arguments, enter the interpreter.
        try {
            fern::enter_interpreter();
            status = EXIT_SUCCESS;
        }
        catch(std::exception const& exception) {
            std::cerr << fern::String(exception.what()) << '\n';
            status = EXIT_FAILURE;
        }
    }
    else if(std::strcmp(argv[1], "--help") == 0) {
        // The help option.
        fern::show_general_help();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--version") == 0) {
        fern::show_version();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--build") == 0) {
        fern::show_build();
        status = EXIT_SUCCESS;
    }
    else {
        std::unique_ptr<fern::Command> command;

        // A command may be given. Find out which one.
        if(std::strcmp(argv[1], "compile") == 0) {
            command.reset(new fern::CompileCommand(argc - 1, argv + 1));
        }
        else if(std::strcmp(argv[1], "describe") == 0) {
            command.reset(new fern::DescribeCommand(argc - 1, argv + 1));
        }
        else if(std::strcmp(argv[1], "execute") == 0) {
            command.reset(new fern::ExecuteCommand(argc - 1, argv + 1));
        }
        else if(std::strcmp(argv[1], "import") == 0) {
            command.reset(new fern::ImportCommand(argc - 1, argv + 1));
        }
        else {
            fern::show_general_help();
            status = EXIT_FAILURE;
        }

        assert(command);

        try {
            status = command->execute();
        }
        catch(std::exception const& exception) {
            std::cerr << fern::String(exception.what()) << '\n';
            status = EXIT_FAILURE;
        }
    }

    return status;
}
