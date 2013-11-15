#include "fern/ast/core/scope_vertex.h"


namespace fern {

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

} // namespace fern
