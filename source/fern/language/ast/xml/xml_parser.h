// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include <string>


namespace fern {
namespace language {

class ModuleVertex;

//! An XmlParser parses an XML document and converts it to a syntax tree.
/*!
  Apart from XML validation checks, no semantic checks are performed. The
  syntax tree has the same semantic content as the XML.

  The parser assumes the XML passed in conforms to the Fern schema.

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

    std::shared_ptr<ModuleVertex> parse_string(
                                        std::string const& xml) const;

private:

    std::shared_ptr<ModuleVertex> parse(std::istream& stream) const;

};

} // namespace language
} // namespace fern
