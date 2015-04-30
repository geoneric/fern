// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/if_vertex.h"


namespace fern {
namespace language {

IfVertex::IfVertex(
    std::shared_ptr<ExpressionVertex> const& condition,
    std::shared_ptr<ScopeVertex> const& true_scope,
    std::shared_ptr<ScopeVertex> const& false_scope)

    : StatementVertex(),
      _condition(condition),
      _true_scope(true_scope),
      _false_scope(false_scope),
      _sentinel(std::make_shared<SentinelVertex>())

{
    assert(_true_scope && !_true_scope->statements().empty());
    assert(_false_scope);
}


std::shared_ptr<ExpressionVertex> const& IfVertex::condition() const
{
    return _condition;
}


std::shared_ptr<ScopeVertex> const& IfVertex::true_scope() const
{
    return _true_scope;
}


std::shared_ptr<ScopeVertex>& IfVertex::true_scope()
{
    return _true_scope;
}


std::shared_ptr<ScopeVertex> const& IfVertex::false_scope() const
{
    return _false_scope;
}


std::shared_ptr<ScopeVertex>& IfVertex::false_scope()
{
    return _false_scope;
}


std::shared_ptr<SentinelVertex> const& IfVertex::sentinel() const
{
    return _sentinel;
}


std::shared_ptr<SentinelVertex>& IfVertex::sentinel()
{
    return _sentinel;
}

} // namespace language
} // namespace fern
