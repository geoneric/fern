#include "geoneric/compiler/module.h"
#include <iostream>


namespace geoneric {

Module::Module(
    std::vector<Argument> const& arguments,
    int argc,
    char** argv)

    : _arguments(arguments),
      _argc(argc),
      _argv(argv)

{
    parse_command_line();
}


//! Parse command line.
/*!
  \exception std::invalid_argument If the command line is not correctly
             formatted.
*/
void Module::parse_command_line()
{
    std::cout << "parse_command_line" << std::endl;
}


//! Run the module.
/*!
  \exception std::runtime_error If an error occured.
*/
void Module::run()
{
}

} // namespace geoneric
