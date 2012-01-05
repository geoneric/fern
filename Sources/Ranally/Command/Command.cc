#include "Command.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "Ranally/Util/String.h"
#include "Ranally/Language/AlgebraParser.h"



namespace ranally {

Command::Command(
  int argc,
  char** argv)

  : _argc(argc),
    _argv(argv)

{
}



Command::~Command()
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



//! Read script from \a fileName.
/*!
  \param     fileName Name of file to read script from.
  \exception .

  In case \a fileName is empty, the script is read from standard input.
*/
UnicodeString Command::read(
  std::string const& fileName)
{
  UnicodeString xml;
  ranally::language::AlgebraParser parser;

  if(fileName.empty()) {
    // Read script from the standard input stream.
    std::ostringstream script;
    script << std::cin.rdbuf();
    xml = parser.parseString(UnicodeString(script.str().c_str()));
  }
  else {
    // Read script from a file.
    xml = parser.parseFile(UnicodeString(fileName.c_str()));
  }

  return xml;
}



//! Write \a contents to a file with name \a fileName.
/*!
  \param     fileName Name of file to write \a contents to.
  \exception .

  In case \a fileName is empty, the \a contents are written to standard output,
  encoded in UTF8.
*/
void Command::write(
  UnicodeString const& contents,
  std::string const& fileName)
{
  if(fileName.empty()) {
    std::cout << ranally::util::encodeInUTF8(contents);
  }
  else {
    std::ofstream file(fileName.c_str());
    file << ranally::util::encodeInUTF8(contents);
  }
}

} // namespace ranally

