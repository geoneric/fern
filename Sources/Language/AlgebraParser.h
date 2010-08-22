#ifndef INCLUDED_RANALLY_ALGEBRAPARSER
#define INCLUDED_RANALLY_ALGEBRAPARSER

#include <unicode/unistr.h>

#include "dev_PythonClient.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AlgebraParser: public dev::PythonClient
{

  friend class AlgebraParserTest;

private:

  // UnicodeString    parse               (UnicodeString const& string,
  //                                       UnicodeString const& fileName);

protected:

public:

                   AlgebraParser       ();

  /* virtual */    ~AlgebraParser      ();

  UnicodeString    parseString         (UnicodeString const& string);

  /// UnicodeString    parseFile           (UnicodeString const& fileName);

};

} // namespace ranally

#endif
