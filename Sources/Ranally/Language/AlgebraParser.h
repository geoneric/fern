#pragma once
#include "Ranally/Python/Client.h"
#include "Ranally/Util/String.h"



namespace ranally {
namespace language {

//! An AlgebraParser parses a script and converts it to XML.
/*!
  AlgebraParser instances only perform syntax checks. If the script is
  syntactically correct, than an XML will be created. This XML has the same
  semantic content as the original script. It's just easier to post process.

  The XML returned conforms to the Ranally.xsd schema.

  \sa        XmlParser
*/
class AlgebraParser:
  public python::Client
{

  friend class AlgebraParserTest;

public:

                   AlgebraParser       ();

                   ~AlgebraParser      ();

  String           parseString         (String const& string);

  String           parseFile           (String const& fileName);

private:

  // String           parse               (String const& string,
  //                                       String const& fileName);

};

} // namespace language
} // namespace ranally
