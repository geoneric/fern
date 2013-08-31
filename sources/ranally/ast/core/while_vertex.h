#pragma once
#include "ranally/ast/core/expression_vertex.h"
#include "ranally/ast/core/scope_vertex.h"
#include "ranally/ast/core/statement_vertex.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class WhileVertex:
    public StatementVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   WhileVertex         (
                        std::shared_ptr<ExpressionVertex> const& condition,
                        std::shared_ptr<ScopeVertex> const& true_scope,
                        std::shared_ptr<ScopeVertex> const& false_scope);

                   ~WhileVertex        ()=default;

                   WhileVertex         (WhileVertex&&)=delete;

    WhileVertex&   operator=           (WhileVertex&&)=delete;

                   WhileVertex         (WhileVertex const&)=delete;

    WhileVertex&   operator=           (WhileVertex const&)=delete;

    std::shared_ptr<ExpressionVertex> const& condition() const;

    std::shared_ptr<ScopeVertex> const& true_scope() const;

    std::shared_ptr<ScopeVertex>& true_scope ();

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

} // namespace ranally
