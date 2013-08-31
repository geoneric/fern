#pragma once
#include "geoneric/ast/core/expression_vertex.h"
#include "geoneric/ast/core/scope_vertex.h"
#include "geoneric/ast/core/statement_vertex.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  Threading an if-statement results in multiple successor vertices being stored in the
  base-class' successor collection.
  - The first successor always points to the first vertex of the true-block's
    control flow graph.
  - If there is a false-block, then the second successor points to the first
    vertex of its control flow graph.
  - The last successor (the second or third, depending on whether or not a
    false-block is present) points to the successor of the whole statement.

  \sa        .
*/
class IfVertex:
    public StatementVertex
{

    friend class IfVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   IfVertex            (
                        std::shared_ptr<ExpressionVertex> const& condition,
                        std::shared_ptr<ScopeVertex> const& true_scope,
                        std::shared_ptr<ScopeVertex> const& false_scope);

                   ~IfVertex           ()=default;

                   IfVertex            (IfVertex&&)=delete;

    IfVertex&      operator=           (IfVertex&&)=delete;

                   IfVertex            (IfVertex const&)=delete;

    IfVertex&      operator=           (IfVertex const&)=delete;

    std::shared_ptr<ExpressionVertex> const& condition() const;

    std::shared_ptr<ScopeVertex> const& true_scope() const;

    std::shared_ptr<ScopeVertex>& true_scope();

    std::shared_ptr<ScopeVertex> const& false_scope() const;

    std::shared_ptr<ScopeVertex>& false_scope();

    std::shared_ptr<SentinelVertex> const& sentinel() const;

    std::shared_ptr<SentinelVertex>& sentinel();

private:

    ExpressionVertexPtr _condition;

    std::shared_ptr<ScopeVertex> _true_scope;

    std::shared_ptr<ScopeVertex> _false_scope;

    std::shared_ptr<SentinelVertex> _sentinel;

};

} // namespace geoneric
