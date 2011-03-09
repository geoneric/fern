#ifndef INCLUDED_RANALLY_LANGUAGE_FUNCTIONVERTEX
#define INCLUDED_RANALLY_LANGUAGE_FUNCTIONVERTEX

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
class FunctionVertex: public ExpressionVertex
{

  friend class FunctionVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

                   FunctionVertex      (UnicodeString const& name,
                                        ExpressionVertices const& expressions);

  /* virtual */    ~FunctionVertex     ();

  ExpressionVertices const& expressions() const;

protected:

private:

  ExpressionVertices _expressions;

};

} // namespace language
} // namespace ranally

#endif
