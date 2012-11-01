#include "DescribeCommand.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Interpreter/Interpreter.h"


namespace ranally {
namespace {

void showDescribeHelp()
{
    std::cout <<
        "usage: ranally describe INPUT_SCRIPT\n"
        "\n"
        "Describe the script.\n"
        "\n"
        "  INPUT_SCRIPT        Script to read or - to read from standard input\n"
        "\n"
        "The result is written to standard output\n"
        ;
}

} // Anonumous namespace


DescribeCommand::DescribeCommand(
    int argc,
    char** argv)

    : Command(argc, argv)

{
}


DescribeCommand::~DescribeCommand()
{
}


void DescribeCommand::describe(
    String const& xml)
{
    std::shared_ptr<ranally::ScriptVertex> tree(
        ranally::XmlParser().parse(xml));
    ranally::Interpreter interpreter;
    interpreter.validate(tree);

    // TODO describe
    // ranally::DescribeVisitor describeVisitor(std::cout);
    // tree->Accept(describeVisitor);
}


int DescribeCommand::execute()
{
    int status = EXIT_FAILURE;

    if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
        // No arguments, or the help option.
        showDescribeHelp();
        status = EXIT_SUCCESS;
    }
    else {
        int currentArgumentId = 1;

        if(argc() - currentArgumentId > 1) {
            std::cerr << "Too many arguments.\n";
            showDescribeHelp();
            status = EXIT_FAILURE;
        }
        else {
            std::string inputFileName =
                std::strcmp(argv()[currentArgumentId], "-") != 0
                    ? argv()[currentArgumentId] : "";
            String xml = read(inputFileName);
            describe(xml);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}

} // namespace ranally
