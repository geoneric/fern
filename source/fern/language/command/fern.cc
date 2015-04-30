// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <cstring>
#include <iostream>
#include <memory>
#include "fern/configure.h"
#include "fern/language/command/compile_command.h"
#include "fern/language/command/describe_command.h"
#include "fern/language/command/execute_command.h"
#include "fern/language/command/import_command.h"
#include "fern/language/command/interpreter.h"
#include "fern/language/command/message.h"


namespace fern {
namespace language {

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
        "  describe            Describe data\n"
        "  import              Import data\n"
        "\n"
        "See 'fern COMMAND --help' for more information on a specific command.\n"
        "The interactive interpreter is entered when no arguments are passed.\n"
        ;
}


void show_build()
{
    std::cout << FERN_BUILD_TYPE << " build (" << __DATE__ << ")\n"
        // TODO Make this FERN_TARGET_ARCHITECTURE.
        // << FERN_ARCHITECTURE << ", "
        << FERN_SYSTEM << ", "
        << FERN_CXX_COMPILER
        << "\n";
}

} // namespace language
} // namespace fern


int main(
    int argc,
    char** argv)
{
    namespace fl = fern::language;

    int status = EXIT_FAILURE;

    if(argc == 1) {
        // No arguments, enter the interpreter.
        try {
            fl::enter_interpreter();
            status = EXIT_SUCCESS;
        }
        catch(std::exception const& exception) {
            std::cerr << fern::String(exception.what()) << '\n';
            status = EXIT_FAILURE;
        }
    }
    else if(std::strcmp(argv[1], "--help") == 0) {
        // The help option.
        fl::show_general_help();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--version") == 0) {
        fl::show_version();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "--build") == 0) {
        fl::show_build();
        status = EXIT_SUCCESS;
    }
    else {
        std::unique_ptr<fl::Command> command;

        // A command may be given. Find out which one.
        if(std::strcmp(argv[1], "compile") == 0) {
            command = std::make_unique<fl::CompileCommand>(argc - 1,
                argv + 1);
        }
        else if(std::strcmp(argv[1], "describe") == 0) {
            command = std::make_unique<fl::DescribeCommand>(argc - 1,
                argv + 1);
        }
        else if(std::strcmp(argv[1], "execute") == 0) {
            command = std::make_unique<fl::ExecuteCommand>(argc - 1,
                argv + 1);
        }
        else if(std::strcmp(argv[1], "import") == 0) {
            command = std::make_unique<fl::ImportCommand>(argc - 1,
                argv + 1);
        }
        else {
            fl::show_general_help();
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
