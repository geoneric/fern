#pragma once
#include "ranally/ast/core/expression_vertex.h"
#include "ranally/ast/core/statement_vertex.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FunctionDefinitionVertex:
    public ExpressionVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   FunctionDefinitionVertex(
                                        String const& name,
                                        ExpressionVertices const& arguments,
                                        StatementVertices const& body);

                   ~FunctionDefinitionVertex()=default;

                   FunctionDefinitionVertex(
                                        FunctionDefinitionVertex&&)=delete;

    FunctionDefinitionVertex& operator=(FunctionDefinitionVertex&&)=delete;

                   FunctionDefinitionVertex(
                                        FunctionDefinitionVertex const&)=delete;

    FunctionDefinitionVertex& operator=(FunctionDefinitionVertex const&)=delete;

    ExpressionVertices const& arguments() const;

    StatementVertices const& body() const;

private:

    ExpressionVertices _arguments;

    StatementVertices _body;

};

} // namespace ranally
