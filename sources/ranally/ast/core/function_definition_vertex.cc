#include "ranally/ast/core/function_definition_vertex.h"


namespace ranally {

FunctionDefinitionVertex::FunctionDefinitionVertex(
    String const& name,
    ExpressionVertices const& arguments,
    StatementVertices const& body)

    : ExpressionVertex(name),
      _arguments(arguments),
      _body(body)

{
}


ExpressionVertices const& FunctionDefinitionVertex::arguments() const
{
    return _arguments;
}


StatementVertices const& FunctionDefinitionVertex::body() const
{
    return _body;
}

} // namespace ranally
