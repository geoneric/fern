#pragma once
#include "Ranally/Language/OperationVertex.h"


namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OperatorVertex:
  public OperationVertex
{

  friend class OperatorVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

                   OperatorVertex      (String const& name,
                                        ExpressionVertices const& expressions);

                   ~OperatorVertex     ();

  String const&    symbol              () const;

private:

  String           _symbol;

};

} // namespace language
} // namespace ranally
