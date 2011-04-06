#include "Ranally/Operation/XmlParser.h"



namespace ranally {
namespace operation {

XmlParser::XmlParser()
{
}



XmlParser::~XmlParser()
{
}



std::map<UnicodeString, Operation_pskel> XmlParser::parse(
  UnicodeString const& /* xml */) const
{
  return std::map<UnicodeString, Operation_pskel>();
}

} // namespace operation
} // namespace ranally

