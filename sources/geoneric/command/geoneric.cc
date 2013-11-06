#include <cstring>
#include <iostream>
#include <memory>
#include "geoneric/configure.h"
#include "geoneric/command/compile_command.h"
#include "geoneric/command/describe_command.h"
#include "geoneric/command/execute_command.h"
#include "geoneric/command/import_command.h"
#include "geoneric/command/interpreter.h"
#include "geoneric/command/message.h"


namespace geoneric {

void show_general_help()
{
    std::cout <<
        "usage: geoneric [--help] [--build] [--version] [COMMAND] [ARGS]\n"
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
        "See 'geoneric COMMAND --help' for more information on a specific command.\n"
        "The interactive interpreter is entered when no arguments are passed.\n"
        ;
}


void show_build()
{
    std::cout << GEONERIC_BUILD_TYPE << " build (" << __DATE__ << ")\n"
        << GEONERIC_ARCHITECTURE << ", "
        << GEONERIC_SYSTEM << ", "
        << GEONERIC_CXX_COMPILER
        << "\n";
}

} // namespace geoneric


int main(
    int argc,
    char** argv)
{
    int status = EXIT_FAILURE;

    if(argc == 1) {
        // No arguments, enter the interpreter.
        try {
            geoneric::enter_interpreter();
            status = EXIT_SUCCESS;
        }
        catch(std::exception const& exception) {
            std::cerr << geoneric::String(exception.what()) << '\n';
            status = EXIT_FAILURE;
        }
    }
    else if(std::strcmp(argv[1], "--help") == 0) {
        // The help option.
        geoneric::show_general_help();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--version") == 0) {
        geoneric::show_version();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--build") == 0) {
        geoneric::show_build();
        status = EXIT_SUCCESS;
    }
    else {
        std::unique_ptr<geoneric::Command> command;

        // A command may be given. Find out which one.
        if(std::strcmp(argv[1], "compile") == 0) {
            command.reset(new geoneric::CompileCommand(argc - 1, argv + 1));
        }
        else if(std::strcmp(argv[1], "describe") == 0) {
            command.reset(new geoneric::DescribeCommand(argc - 1, argv + 1));
        }
        else if(std::strcmp(argv[1], "execute") == 0) {
            command.reset(new geoneric::ExecuteCommand(argc - 1, argv + 1));
        }
        else if(std::strcmp(argv[1], "import") == 0) {
            command.reset(new geoneric::ImportCommand(argc - 1, argv + 1));
        }
        else {
            geoneric::show_general_help();
            status = EXIT_FAILURE;
        }

        assert(command);

        try {
            status = command->execute();
        }
        catch(std::exception const& exception) {
            std::cerr << geoneric::String(exception.what()) << '\n';
            status = EXIT_FAILURE;
        }
    }

    return status;
}
