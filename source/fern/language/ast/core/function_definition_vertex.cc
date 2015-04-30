// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/function_definition_vertex.h"


namespace fern {
namespace language {

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

} // namespace language
} // namespace fern
