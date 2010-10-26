#ifndef INCLUDED_RANALLY_STATEMENTVERTEX
#define INCLUDED_RANALLY_STATEMENTVERTEX

#include "SyntaxVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class StatementVertex: public SyntaxVertex
{

  friend class StatementVertexTest;

private:

protected:

                   StatementVertex     ();

                   StatementVertex     (int lineNr,
                                        int colId);

public:

  LOKI_DEFINE_VISITABLE()

  virtual          ~StatementVertex    ();

};

} // namespace ranally

#endif
