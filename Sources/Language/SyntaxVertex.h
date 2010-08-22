#ifndef INCLUDED_RANALLY_SYNTAXVERTEX
#define INCLUDED_RANALLY_SYNTAXVERTEX

#include <unicode/unistr.h>



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SyntaxVertex
{

  friend class SyntaxVertexTest;

private:
  int              _line;

  int              _col;

  UnicodeString    _value;

protected:

public:

                   SyntaxVertex        (int lineNr,
                                        int colId,
                                        UnicodeString const& value);

  virtual          ~SyntaxVertex       ();

};

} // namespace ranally

#endif
