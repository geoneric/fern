#include "ranally/command/describe_command.h"


namespace ranally {
namespace {

void show_describe_help()
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


DescribeCommand::~DescribeCommand() noexcept(true) =default;


void DescribeCommand::describe(
    ScriptVertexPtr const& tree) const
{
    interpreter().validate(tree);

    // TODO describe
    // ranally::DescribeVisitor describe_visitor(std::cout);
    // tree->Accept(describe_visitor);
}


int DescribeCommand::execute() const
{
    int status = EXIT_FAILURE;

    if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
        // No arguments, or the help option.
        show_describe_help();
        status = EXIT_SUCCESS;
    }
    else {
        int current_argument_id = 1;

        if(argc() - current_argument_id > 1) {
            std::cerr << "Too many arguments.\n";
            show_describe_help();
            status = EXIT_FAILURE;
        }
        else {
            std::string input_filename =
                std::strcmp(argv()[current_argument_id], "-") != 0
                    ? argv()[current_argument_id] : "";
            ScriptVertexPtr tree(interpreter().parse_file(input_filename));
            describe(tree);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}

} // namespace ranally
