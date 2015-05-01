// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/while_vertex.h"
#include <cassert>


namespace fern {
namespace language {

WhileVertex::WhileVertex(
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


std::shared_ptr<ExpressionVertex> const& WhileVertex::condition() const
{
    return _condition;
}


std::shared_ptr<ScopeVertex> const& WhileVertex::true_scope() const
{
    return _true_scope;
}


std::shared_ptr<ScopeVertex>& WhileVertex::true_scope()
{
    return _true_scope;
}


std::shared_ptr<ScopeVertex> const& WhileVertex::false_scope() const
{
    return _false_scope;
}


std::shared_ptr<ScopeVertex>& WhileVertex::false_scope()
{
    return _false_scope;
}


std::shared_ptr<SentinelVertex> const& WhileVertex::sentinel() const
{
    return _sentinel;
}


std::shared_ptr<SentinelVertex>& WhileVertex::sentinel()
{
    return _sentinel;
}

} // namespace language
} // namespace fern
