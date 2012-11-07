#pragma once
#include <memory>
#include "Ranally/Util/String.h"


namespace ranally {

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

public:

                   XmlParser           ()=default;

                   ~XmlParser          ()=default;

                   XmlParser           (XmlParser&&)=delete;

    XmlParser&     operator=           (XmlParser&&)=delete;

                   XmlParser           (XmlParser const&)=delete;

    XmlParser&     operator=           (XmlParser const&)=delete;

    std::shared_ptr<ScriptVertex> parse(String const& xml) const;

    std::shared_ptr<ScriptVertex> parse(std::istream& stream) const;

};

} // namespace ranally
