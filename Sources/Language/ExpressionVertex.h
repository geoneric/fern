#ifndef INCLUDED_RANALLY_EXPRESSIONVERTEX
#define INCLUDED_RANALLY_EXPRESSIONVERTEX

#include "StatementVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ExpressionVertex: public StatementVertex
{

  friend class ExpressionVertexTest;

private:

protected:

                   ExpressionVertex    ();

                   ExpressionVertex    (int lineNr,
                                        int colId);

public:

  virtual          ~ExpressionVertex   ();

};

} // namespace ranally

#endif
