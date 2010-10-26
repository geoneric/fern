#ifndef INCLUDED_RANALLY_FUNCTIONVERTEX
#define INCLUDED_RANALLY_FUNCTIONVERTEX

#include <vector>
#include <boost/shared_ptr.hpp>
#include <unicode/unistr.h>

#include "ExpressionVertex.h"



namespace ranally {

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

private:

  UnicodeString    _name;

  ExpressionVertices _expressions;

protected:

public:

                   FunctionVertex      (UnicodeString const& name,
                                        ExpressionVertices const& expressions);

  /* virtual */    ~FunctionVertex     ();

  UnicodeString const& name            () const;

  ExpressionVertices const& expressions() const;

};

} // namespace ranally

#endif
