#include "geoneric/command/import_command.h"
#include <cstring>
#include <iostream>


namespace geoneric {
namespace {

void show_import_help()
{
    std::cout <<
        "usage: geoneric import INPUT_DATA_SET OUTPUT_DATA_SET\n"
        "\n"
        "Import the data set.\n"
        "\n"
        "  INPUT_DATA_SET      Data set to import\n"
        "  OUTPUT_DATA_SET     Data set to write\n"
        ;
}

} // Anonymous namespace


ImportCommand::ImportCommand(
    int argc,
    char** argv)

    : Command(argc, argv)

{
}


int ImportCommand::execute() const
{
    int status = EXIT_FAILURE;

    if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
        // No arguments, or the help option.
        show_import_help();
        status = EXIT_SUCCESS;
    }
    else if(argc() == 2) {
        std::cerr << "Not enough arguments.\n";
        show_import_help();
        status = EXIT_FAILURE;
    }
    else if(argc() > 3) {
        std::cerr << "Too many arguments.\n";
        show_import_help();
        status = EXIT_FAILURE;
    }
    else {
        String input_dataset_name = argv()[1];
        String output_dataset_name = argv()[2];
        // import(input_dataset_name, output_dataset_name);
        status = EXIT_SUCCESS;
    }

    return status;
}

} // namespace geoneric
