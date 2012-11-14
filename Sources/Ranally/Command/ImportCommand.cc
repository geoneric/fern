#include "ImportCommand.h"
#include <cstring>
#include <iostream>
#include "Ranally/IO/Util.h"


namespace ranally {
namespace {

void showImportHelp()
{
    std::cout <<
        "usage: ranally import INPUT_DATA_SET OUTPUT_DATA_SET\n"
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


int ImportCommand::execute()
{
    int status = EXIT_FAILURE;

    if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
        // No arguments, or the help option.
        showImportHelp();
        status = EXIT_SUCCESS;
    }
    else if(argc() == 2) {
        std::cerr << "Not enough arguments.\n";
        showImportHelp();
        status = EXIT_FAILURE;
    }
    else if(argc() > 3) {
        std::cerr << "Too many arguments.\n";
        showImportHelp();
        status = EXIT_FAILURE;
    }
    else {
        String inputDatasetName = argv()[1];
        String outputDatasetName = argv()[2];
        ranally::import(inputDatasetName, outputDatasetName);
        status = EXIT_SUCCESS;
    }

    return status;
}

} // namespace ranally
