#include "ranally/ast/core/scope_vertex.h"


namespace ranally {

ScopeVertex::ScopeVertex(
    StatementVertices const& statements)

    : AstVertex(),
      _statements(statements),
      _sentinel(new SentinelVertex())

{
}


StatementVertices const& ScopeVertex::statements() const
{
    return _statements;
}


StatementVertices& ScopeVertex::statements()
{
    return _statements;
}


std::shared_ptr<SentinelVertex> const& ScopeVertex::sentinel() const
{
    return _sentinel;
}


std::shared_ptr<SentinelVertex>& ScopeVertex::sentinel()
{
    return _sentinel;
}

} // namespace ranally
