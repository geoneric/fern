#include "fern/ast/core/while_vertex.h"


namespace fern {

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

} // namespace fern
