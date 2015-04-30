// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/command/describe_command.h"
#include "fern/core/data_name.h"
#include "fern/language/io/drivers.h"


namespace fern {
namespace language {
namespace {

void show_describe_help()
{
    std::cout <<
        "usage: fern describe INPUT_DATA\n"
        "\n"
        "Describe the data.\n"
        "\n"
        "  INPUT_DATA          Data to describe\n"
        "\n"
        "The result is written to standard output\n"
        ;
}

} // Anonumous namespace


DescribeCommand::DescribeCommand(
    int argc,
    char** argv)

    : IOClient(),
      Command(argc, argv)

{
}


void DescribeCommand::describe_feature(
    Dataset const& dataset,
    Path const& path) const
{
    auto feature = dataset.open_feature(path);
    std::cout << "feature: " << path << std::endl;
    // TODO
}


void DescribeCommand::describe_attribute(
    Dataset const& dataset,
    Path const& path) const
{
    auto attribute = dataset.open_attribute(path);
    std::cout << "attribute: " << path << std::endl;
    // TODO Hier verder.
}


void DescribeCommand::describe(
    DataName const& data_name) const
{
    if(!dataset_exists(data_name.database_pathname().generic_string(),
            OpenMode::READ)) {
        std::cout << "Dataset does not exist\n";
    }
    else {
        auto dataset = open_dataset(
            data_name.database_pathname().generic_string(), OpenMode::READ);
        assert(dataset);

        if(!data_name.data_pathname().is_empty()) {
            // Describe data pointed to by pathname.
            if(dataset->contains_feature(data_name.data_pathname())) {
                // Describe feature.
                describe_feature(*dataset, data_name.data_pathname());
            }
            else if(dataset->contains_attribute(data_name.data_pathname())) {
                // Describe attribute.
                describe_attribute(*dataset, data_name.data_pathname());
            }
            else {
                std::cout
                    << "Datapath does not point to a feature or attribute\n";
            }
        }
        else {
            // Describe whole database.
            for(auto const& feature_name: dataset->feature_names()) {
                // Describe feature.
                describe_feature(*dataset, feature_name);
            }
        }
    }
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
            describe(argv()[current_argument_id]);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}

} // namespace language
} // namespace fern
