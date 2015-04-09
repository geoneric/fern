// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/command/execute_command.h"


namespace fern {
namespace {

void show_execute_help()
{
    std::cout <<
        "usage: fern execute [--help] INPUT_SCRIPT\n"
        "\n"
        "Execute the script.\n"
        "\n"
        "  INPUT_SCRIPT        Script to execute or - to read from standard input\n"
        ;
}

} // Anonymous namespace


ExecuteCommand::ExecuteCommand(
    int argc,
    char** argv)

    : IOClient(),
      Command(argc, argv)

{
}


void ExecuteCommand::execute(
    ModuleVertexPtr const& tree) const
{
    const_cast<Interpreter&>(interpreter()).execute(tree);
}


int ExecuteCommand::execute() const
{
    int status = EXIT_FAILURE;

    if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
        // No arguments, or the help option.
        show_execute_help();
        status = EXIT_SUCCESS;
    }
    else if(argc() > 2) {
        std::cerr << "Too many arguments.\n";
        show_execute_help();
        status = EXIT_FAILURE;
    }
    else {
        std::string input_filename = std::strcmp(argv()[1], "-") != 0
            ? argv()[1] : "";
        ModuleVertexPtr tree(interpreter().parse_file(input_filename));
        execute(tree);
        status = EXIT_SUCCESS;
    }

    return status;
}

} // namespace fern
