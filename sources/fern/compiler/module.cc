#include "fern/compiler/module.h"
#include <sstream>
#include <boost/program_options.hpp>


namespace fern {
namespace {

//! Parse command line.
/*!
  \exception std::invalid_argument If the command line is not correctly
             formatted.
*/
std::vector<std::shared_ptr<DataSource>> parse_command_line(
    std::vector<DataDescription> const& arguments,
    int argc,
    char** argv)
{
    namespace po = boost::program_options;

    po::options_description options; // ("Allowed options");
    // options.add_options()
    //     ("help", "produce help message")
    //     ;

    po::positional_options_description positional_options;
    for(auto argument: arguments) {
        std::string name(argument.name());
        options.add_options()(name.c_str(), argument.description().c_str());
        positional_options.add(name.c_str(), 1);
    }

    po::variables_map variables_map;
    po::store(po::command_line_parser(argc, argv)
        .options(options)
        .positional(positional_options)
        .run(), variables_map);
    po::notify(variables_map);

    // if(variables_map.count("help")) {
    //     std::cout << options << "\n";
    //     throw std::invalid_argument("
    // }

    for(auto argument: arguments) {
        if(variables_map.count(argument.name())) {
            std::cout << argument.name() << ": "
                << variables_map[argument.name()].as<std::string>() << "\n";
        }
        else {
            std::ostringstream stream;
            stream << options;
            throw std::invalid_argument(stream.str().c_str());
        }
    }

    std::vector<std::shared_ptr<DataSource>> data_sources;

    return data_sources;
}

} // Anonymous namespace


Module::Module(
    std::vector<DataDescription> const& arguments,
    std::vector<std::shared_ptr<DataSource>> const& data_sources)

    : _arguments(arguments),
      _data_sources(data_sources)

{
    if(_arguments.size() > _data_sources.size()) {
        // TODO Message.
        throw std::invalid_argument("Not enough data sources");
    }
    else if(_arguments.size() < _data_sources.size()) {
        // TODO Message.
        throw std::invalid_argument("Too many data sources");
    }
}


Module::Module(
    std::vector<DataDescription> const& arguments,
    int argc,
    char** argv)

    : Module{arguments, parse_command_line(arguments, argc, argv)}

{
}


//! Run the module.
/*!
  \exception std::runtime_error If an error occured.
*/
void Module::run()
{
}

} // namespace fern
