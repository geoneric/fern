// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/scope_vertex.h"


namespace fern {
namespace language {

ScopeVertex::ScopeVertex(
    StatementVertices const& statements)

    : AstVertex(),
      _statements(statements),
      _sentinel(std::make_shared<SentinelVertex>())

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

} // namespace language
} // namespace fern
