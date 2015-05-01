// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/compiler/parse_command_line.h"
#include "fern/core/string.h"
#include "fern/language/io/drivers.h"
#include "fern/language/interpreter/data_sources.h"
#include "fern/language/interpreter/data_syncs.h"


namespace fern {
namespace language {
namespace {

/*!
    @brief      Return the data source corresponding with the @a value
                passed in.
    @param      argument
    @param      value Data name or value of data to use as source. This
                name can point to a feature, domain, attribute, ...
*/
std::shared_ptr<DataSource> data_source(
    DataDescription const& /* argument */,
    std::string const& value)
{
    // First, check if the string passed in is a data name pointing to
    // existing data.
    // If not, determine the value's type and treat it as a constant.
    std::shared_ptr<DataSource> result;

    // If the dataset exists, it is assumed that the data path exists too.
    // If not, than that is an error. We don't need to test the data path
    // here.
    DataName data_name(value);
    if(dataset_exists(data_name.database_pathname().generic_string(),
            OpenMode::READ)) {
        result = std::make_shared<DatasetSource>(data_name);
    }

    if(is_convertable_to<int64_t>(value)) {
        result = std::make_shared<ConstantSource<int64_t>>(as<int64_t>(value));
    }
    else if(is_convertable_to<double>(value)) {
        result = std::make_shared<ConstantSource<double>>(as<double>(value));
    }
    else {
        result = std::make_shared<ConstantSource<std::string>>(value);
    }

    assert(result);
    return result;
}


std::shared_ptr<DataSync> data_sync(
    DataDescription const& /* argument */,
    std::string const& value)
{
    // The value passed in must be the name of a data sync that can be opened.
    return std::shared_ptr<DataSync>(std::make_shared<DatasetSync>(
        DataName(value)));
}

} // Anonymous namespace


/*!
    @brief      Parse the command line arguments and return collections
                of data sources and syncs.
    @param      argc Argument count.
    @param      argv Argument vector.
    @param      arguments Properties of arguments.
    @param      arguments Properties of results.
    @exception  std::invalid_argument When there are more command line
                arguments passed in compared to the @a arguments and @a
                results passed in.

  For each argument in @a arguments, a command line argument is searched. This
  command line argument is converted to a data source.

  For each result in @a results, a command line argument searched. This command
  line argument is converted to a data sync.

  This function does not try to check whether there are enough arguments and
  whether data sources are valid, exist, etc. That is the job of the caller.
*/
std::tuple<
    std::vector<std::shared_ptr<DataSource>>,
    std::vector<std::shared_ptr<DataSync>>>
parse_command_line(
    int argc,
    char** argv,
    std::vector<DataDescription> const& arguments,
    std::vector<DataDescription> const& results)
{
    assert(argc >= 1);  // First argument is the exe's name.
    size_t nr_remaining_command_line_arguments = argc - 1;
    char** current_command_line_argument = argv + 1;

    // Parse command line.
    // - Start with creating data sources.
    // - Continue creating data syncs.

    std::vector<std::shared_ptr<DataSource>> data_sources;
    std::vector<std::shared_ptr<DataSync>> data_syncs;

    for(auto argument: arguments) {
        if(nr_remaining_command_line_arguments == 0u) {
            break;
        }

        data_sources.emplace_back(data_source(argument,
            *current_command_line_argument));
        --nr_remaining_command_line_arguments;
        ++current_command_line_argument;
    }

    for(auto result: results) {
        if(nr_remaining_command_line_arguments == 0u) {
            break;
        }

        data_syncs.emplace_back(data_sync(result,
            *current_command_line_argument));
        --nr_remaining_command_line_arguments;
        ++current_command_line_argument;
    }

    if(nr_remaining_command_line_arguments > 0u) {
        // TODO Message.
        throw std::invalid_argument("Too many command line arguments provided");
    }

    return std::make_tuple(data_sources, data_syncs);
}

} // namespace language
} // namespace fern
