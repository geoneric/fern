#ifndef INCLUDED_RANALLY_NAMEVERTEX
#define INCLUDED_RANALLY_NAMEVERTEX

#include <unicode/unistr.h>

#include "ExpressionVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class NameVertex: public ExpressionVertex
{

  friend class NameVertexTest;

private:

  UnicodeString    _name;

protected:

public:

  LOKI_DEFINE_VISITABLE()

                   NameVertex          (int lineNr,
                                        int colId,
                                        UnicodeString const& name);

  /* virtual */    ~NameVertex         ();

  UnicodeString const& name            () const;

};

} // namespace ranally

#endif
