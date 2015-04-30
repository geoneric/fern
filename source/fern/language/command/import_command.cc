// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/command/import_command.h"
#include <cstring>
#include <iostream>


namespace fern {
namespace language {
namespace {

void show_import_help()
{
    std::cout <<
        "usage: fern import INPUT_DATA_SET OUTPUT_DATA_SET\n"
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

} // namespace language
} // namespace fern
