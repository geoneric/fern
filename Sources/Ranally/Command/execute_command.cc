#include "Ranally/Command/execute_command.h"
#include "Ranally/Language/xml_parser.h"
#include "Ranally/Interpreter/interpreter.h"


namespace ranally {
namespace {

void showExecuteHelp()
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
    String const& xml)
{
    std::shared_ptr<ranally::ScriptVertex> tree(
        ranally::XmlParser().parse(xml));
    ranally::Interpreter interpreter;
    interpreter.execute(tree);
}


int ExecuteCommand::execute()
{
    int status = EXIT_FAILURE;

    if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
        // No arguments, or the help option.
        showExecuteHelp();
        status = EXIT_SUCCESS;
    }
    else if(argc() > 2) {
        std::cerr << "Too many arguments.\n";
        showExecuteHelp();
        status = EXIT_FAILURE;
    }
    else {
        std::string inputFileName = std::strcmp(argv()[1], "-") != 0
            ? argv()[1] : "";
        String xml = read(inputFileName);
        execute(xml);
        status = EXIT_SUCCESS;
    }

    return status;
}

} // namespace ranally
