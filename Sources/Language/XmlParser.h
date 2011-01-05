#ifndef INCLUDED_RANALLY_XMLPARSER
#define INCLUDED_RANALLY_XMLPARSER

#include <unicode/unistr.h>
#include <boost/shared_ptr.hpp>



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

protected:

public:

                   XmlParser           ();

  /* virtual */    ~XmlParser          ();

  boost::shared_ptr<ScriptVertex> parse(UnicodeString const& xml) const;

  boost::shared_ptr<ScriptVertex> parse(std::istream& stream) const;

};

} // namespace language
} // namespace ranally

#endif
