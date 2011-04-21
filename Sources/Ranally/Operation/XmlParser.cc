#include "Ranally/Operation/XmlParser.h"

#include <sstream>
#include "dev_UnicodeUtils.h"



namespace ranally {
namespace operation {

XmlParser::XmlParser()
{
}



XmlParser::~XmlParser()
{
}



std::map<UnicodeString, Operation_pskel> XmlParser::parse(
  std::istream& stream) const
{
  return std::map<UnicodeString, Operation_pskel>();
}



std::map<UnicodeString, Operation_pskel> XmlParser::parse(
  UnicodeString const& xml) const
{
  // Copy string contents in a string stream and work with that.
  std::stringstream stream;
  stream.exceptions(std::ifstream::badbit | std::ifstream::failbit);
  stream << dev::encodeInUTF8(xml); // << std::endl;

  return parse(stream);
}

} // namespace operation
} // namespace ranally

