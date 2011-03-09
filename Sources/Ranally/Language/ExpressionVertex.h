#ifndef INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEX
#define INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEX

#include "Ranally/Language/StatementVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ExpressionVertex: public StatementVertex
{

  friend class ExpressionVertexTest;

public:

  virtual          ~ExpressionVertex   ();

  UnicodeString const& name            () const;

protected:

                   ExpressionVertex    (UnicodeString const& name);

                   ExpressionVertex    (int lineNr,
                                        int colId,
                                        UnicodeString const& name);

private:

  UnicodeString    _name;

};

} // namespace language
} // namespace ranally

#endif
