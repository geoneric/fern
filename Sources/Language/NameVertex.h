#ifndef INCLUDED_RANALLY_NAMEVERTEX
#define INCLUDED_RANALLY_NAMEVERTEX

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

protected:

public:

                   NameVertex          (int lineNr,
                                        int colId,
                                        UnicodeString const& name);

  /* virtual */    ~NameVertex         ();

};

} // namespace ranally

#endif
