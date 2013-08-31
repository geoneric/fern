#include "ranally/command/command.h"
#include <fstream>
#include "ranally/script/algebra_parser.h"


namespace ranally {

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
// ranally::String Command::read(
//     std::string const& filename)
// {
//     ranally::String xml;
//     ranally::AlgebraParser parser;
// 
//     if(filename.empty()) {
//         // Read script from the standard input stream.
//         std::ostringstream script;
//         script << std::cin.rdbuf();
//         xml = parser.parseString(ranally::String(script.str()));
//     }
//     else {
//         // Read script from a file.
//         xml = parser.parseFile(ranally::String(filename));
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
    ranally::String const& contents,
    std::string const& filename) const
{
    if(filename.empty()) {
        std::cout << contents.encode_in_utf8();
    }
    else {
        std::ofstream file(filename.c_str());
        file << contents.encode_in_utf8();
    }
}

} // namespace ranally
