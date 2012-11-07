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

                   ~FunctionVertex     ()=default;

                   FunctionVertex      (FunctionVertex&&)=delete;

    FunctionVertex& operator=          (FunctionVertex&&)=delete;

                   FunctionVertex      (FunctionVertex const&)=delete;

    FunctionVertex& operator=          (FunctionVertex const&)=delete;

private:

};

} // namespace ranally
