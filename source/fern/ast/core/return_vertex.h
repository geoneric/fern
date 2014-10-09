#pragma once
#include "fern/ast/core/expression_vertex.h"
#include "fern/ast/core/statement_vertex.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ReturnVertex:
    public StatementVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   ReturnVertex        ();

                   ReturnVertex        (ExpressionVertexPtr const& expression);

                   ~ReturnVertex       ()=default;

                   ReturnVertex        (ReturnVertex&&)=delete;

    ReturnVertex&  operator=           (ReturnVertex&&)=delete;

                   ReturnVertex        (ReturnVertex const&)=delete;

    ReturnVertex&  operator=           (ReturnVertex const&)=delete;

    ExpressionVertexPtr const& expression() const;

private:

    ExpressionVertexPtr _expression;

};

} // namespace fern
