#ifndef INCLUDED_RANALLY_STRINGVERTEX
#define INCLUDED_RANALLY_STRINGVERTEX

#include <unicode/unistr.h>

#include "ExpressionVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class StringVertex: public ExpressionVertex
{

  friend class StringVertexTest;

private:

  UnicodeString    _string;

protected:

public:

                   StringVertex        (int lineNr,
                                        int colId,
                                        UnicodeString const& string);

  /* virtual */    ~StringVertex       ();

};

} // namespace ranally

#endif
