#ifndef INCLUDED_RANALLY_LANGUAGE_XMLPARSER
#define INCLUDED_RANALLY_LANGUAGE_XMLPARSER

#include <boost/shared_ptr.hpp>
#include "Ranally/Util/String.h"



namespace ranally {
namespace language {

class ScriptVertex;

//! An XmlParser parses an Xml and converts it to a syntax tree.
/*!
  Apart from Xml validation checks, no semantic checks are performed. The
  syntax tree has the same semantic content as the Xml.

  The parser assumes the Xml passed in conforms to the Ranally.xsd schema.

  \sa        AlgebraParser
*/
class XmlParser
{

  friend class XmlParserTest;

private:

public:

                   XmlParser           ();

                   ~XmlParser          ();

  boost::shared_ptr<ScriptVertex> parse(String const& xml) const;

  boost::shared_ptr<ScriptVertex> parse(std::istream& stream) const;

};

} // namespace language
} // namespace ranally

#endif
