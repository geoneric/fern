#include "ranally/command/execute_command.h"


namespace ranally {
namespace {

void show_execute_help()
{
    std::cout <<
        "usage: ranally execute [--help] INPUT_SCRIPT\n"
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

    : Command(argc, argv)

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

} // namespace ranally
