#include "ranally/ast/core/function_definition_vertex.h"


namespace ranally {

FunctionDefinitionVertex::FunctionDefinitionVertex(
    String const& name,
    ExpressionVertices const& arguments,
    StatementVertices const& body)

    : StatementVertex(),
      _name(name),
      _arguments(arguments),
      _body(body)

{
}


String const& FunctionDefinitionVertex::name() const
{
    return _name;
}


ExpressionVertices const& FunctionDefinitionVertex::arguments() const
{
    return _arguments;
}


ExpressionVertices& FunctionDefinitionVertex::arguments()
{
    return _arguments;
}


StatementVertices const& FunctionDefinitionVertex::body() const
{
    return _body;
}


StatementVertices& FunctionDefinitionVertex::body()
{
    return _body;
}

} // namespace ranally
