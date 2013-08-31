#include "geoneric/ast/core/function_definition_vertex.h"


namespace geoneric {

FunctionDefinitionVertex::FunctionDefinitionVertex(
    String const& name,
    ExpressionVertices const& arguments,
    std::shared_ptr<ScopeVertex> const& scope)

    : StatementVertex(),
      _name(name),
      _arguments(arguments),
      _scope(scope)

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


std::shared_ptr<ScopeVertex> const& FunctionDefinitionVertex::scope() const
{
    return _scope;
}


std::shared_ptr<ScopeVertex>& FunctionDefinitionVertex::scope()
{
    return _scope;
}

} // namespace geoneric
