#include "fern/compiler/module.h"


namespace fern {

Module::Module(
    std::vector<DataDescription> const& arguments,
    std::vector<DataDescription> const& results)

    : _arguments(arguments),
      _results(results)

{
}


std::vector<DataDescription> const& Module::arguments() const
{
    return _arguments;
}


std::vector<DataDescription> const& Module::results() const
{
    return _results;
}


/*!
  \exception std::runtime_error If an error occured.
*/
void Module::check_sources_and_syncs(
    std::vector<std::shared_ptr<DataSource>> const& data_sources,
    std::vector<std::shared_ptr<DataSync>> const& /* data_syncs */) const
{
    if(_arguments.size() > data_sources.size()) {
        // TODO Message.
        throw std::invalid_argument("Not enough data sources");
    }
    else if(_arguments.size() < data_sources.size()) {
        // TODO Message.
        throw std::invalid_argument("Too many data sources");
    }

    // TODO Compare properties of arguments with properties of data sources.
}

} // namespace fern
