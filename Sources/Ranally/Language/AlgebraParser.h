#ifndef INCLUDED_RANALLY_LANGUAGE_ALGEBRAPARSER
#define INCLUDED_RANALLY_LANGUAGE_ALGEBRAPARSER

#include <unicode/unistr.h>
#include "dev_PythonClient.h"



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
  public dev::PythonClient
{

  friend class AlgebraParserTest;

public:

                   AlgebraParser       ();

                   ~AlgebraParser      ();

  UnicodeString    parseString         (UnicodeString const& string);

  UnicodeString    parseFile           (UnicodeString const& fileName);

private:

  // UnicodeString    parse               (UnicodeString const& string,
  //                                       UnicodeString const& fileName);

};

} // namespace language
} // namespace ranally

#endif
