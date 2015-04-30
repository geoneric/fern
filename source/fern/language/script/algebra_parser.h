// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/string.h"
#include "fern/language/python/client.h"


namespace fern {
namespace language {

//! An AlgebraParser parses a script and converts it to XML.
/*!
  AlgebraParser instances only perform syntax checks. If the script is
  syntactically correct, than an XML will be created. This XML has the same
  semantic content as the original script. It's just easier to post process.

  The XML returned conforms to the Fern schema.

  \sa        XmlParser
  \todo      The arena used by the parse functions could be a member variable.
             Currently the arena is created whenever a parse function is
             called. What makes more sense?
*/
class AlgebraParser:
    public python::Client
{

    friend class AlgebraParserTest;

public:

                   AlgebraParser       ();

                   ~AlgebraParser      ()=default;

                   AlgebraParser       (AlgebraParser&&)=delete;

    AlgebraParser& operator=           (AlgebraParser&&)=delete;

                   AlgebraParser       (AlgebraParser const&)=delete;

    AlgebraParser& operator=           (AlgebraParser const&)=delete;

    String         parse_string        (String const& string) const;

    String         parse_file          (String const& filename) const;

private:

};

} // namespace language
} // namespace fern
