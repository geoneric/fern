// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/command/command.h"
#include <fstream>
#include "fern/language/script/algebra_parser.h"


namespace fern {
namespace language {

Command::Command(
    int argc,
    char** argv)

    : _argc(argc),
      _argv(argv),
      _interpreter()

{
}


int Command::argc() const
{
    return _argc;
}


char** Command::argv() const
{
    return _argv;
}


Interpreter const& Command::interpreter() const
{
    return _interpreter;
}


// //! Read script from \a filename.
// /*!
//   \param     filename Name of file to read script from.
//   \exception .
// 
//   In case \a filename is empty, the script is read from standard input.
// */
// std::string Command::read(
//     std::string const& filename)
// {
//     std::string xml;
//     fern::AlgebraParser parser;
// 
//     if(filename.empty()) {
//         // Read script from the standard input stream.
//         std::ostringstream script;
//         script << std::cin.rdbuf();
//         xml = parser.parseString(script.str());
//     }
//     else {
//         // Read script from a file.
//         xml = parser.parseFile(filename);
//     }
// 
//     return xml;
// }


//! Write \a contents to a file with name \a filename.
/*!
  \param     filename Name of file to write \a contents to.
  \exception .

  In case \a filename is empty, the \a contents are written to standard
  output, encoded in UTF8.
*/
void Command::write(
    std::string const& contents,
    std::string const& filename) const
{
    if(filename.empty()) {
        std::cout << contents;
    }
    else {
        std::ofstream file(filename.c_str());
        file << contents;
    }
}

} // namespace language
} // namespace fern
