#ifndef INCLUDED_RANALLY_LANGUAGE_OPERATORVERTEX
#define INCLUDED_RANALLY_LANGUAGE_OPERATORVERTEX

#include <vector>
#include <boost/shared_ptr.hpp>
#include "Ranally/Language/ExpressionVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OperatorVertex: public ExpressionVertex
{

  friend class OperatorVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

                   OperatorVertex      (UnicodeString const& name,
                                        ExpressionVertices const& expressions);

  /* virtual */    ~OperatorVertex     ();

  UnicodeString const& symbol          () const;

  ExpressionVertices const& expressions() const;

protected:

private:

  UnicodeString    _symbol;

  ExpressionVertices _expressions;

};

} // namespace language
} // namespace ranally

#endif
