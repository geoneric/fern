#ifndef INCLUDED_RANALLY_LANGUAGE_STATEMENTVERTEX
#define INCLUDED_RANALLY_LANGUAGE_STATEMENTVERTEX

#include "Ranally/Language/SyntaxVertex.h"



namespace ranally {
namespace language {

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

} // namespace language
} // namespace ranally

#endif
