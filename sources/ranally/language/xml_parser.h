#pragma once
#include <memory>
#include "ranally/core/string.h"


namespace ranally {

class ScriptVertex;

//! An XmlParser parses an XML document and converts it to a syntax tree.
/*!
  Apart from XML validation checks, no semantic checks are performed. The
  syntax tree has the same semantic content as the XML.

  The parser assumes the XML passed in conforms to the Ranally.xsd schema.

  \sa        AlgebraParser
*/
class XmlParser
{

    friend class XmlParserTest;

public:

                   XmlParser           ()=default;

                   ~XmlParser          ()=default;

                   XmlParser           (XmlParser&&)=delete;

    XmlParser&     operator=           (XmlParser&&)=delete;

                   XmlParser           (XmlParser const&)=delete;

    XmlParser&     operator=           (XmlParser const&)=delete;

    std::shared_ptr<ScriptVertex> parse(std::istream& stream) const;

    std::shared_ptr<ScriptVertex> parse(String const& xml) const;

};

} // namespace ranally
