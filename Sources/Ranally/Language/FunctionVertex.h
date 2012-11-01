#pragma once
#include "Ranally/Language/OperationVertex.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FunctionVertex:
    public OperationVertex
{

    friend class FunctionVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   FunctionVertex      (String const& name,
                                        ExpressionVertices const& expressions);

                   ~FunctionVertex     ();

private:

};

} // namespace ranally
