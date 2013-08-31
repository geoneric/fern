#pragma once
#include "geoneric/core/string.h"
#include "geoneric/python/client.h"


namespace geoneric {

//! An AlgebraParser parses a script and converts it to XML.
/*!
  AlgebraParser instances only perform syntax checks. If the script is
  syntactically correct, than an XML will be created. This XML has the same
  semantic content as the original script. It's just easier to post process.

  The XML returned conforms to the Geoneric.xsd schema.

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

} // namespace geoneric
