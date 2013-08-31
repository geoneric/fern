#pragma once
#include "geoneric/ast/core/operation_vertex.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FunctionCallVertex:
    public OperationVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   FunctionCallVertex  (String const& name,
                                        ExpressionVertices const& expressions);

                   ~FunctionCallVertex ()=default;

                   FunctionCallVertex  (FunctionCallVertex&&)=delete;

    FunctionCallVertex& operator=      (FunctionCallVertex&&)=delete;

                   FunctionCallVertex  (FunctionCallVertex const&)=delete;

    FunctionCallVertex& operator=      (FunctionCallVertex const&)=delete;

private:

};

} // namespace geoneric
