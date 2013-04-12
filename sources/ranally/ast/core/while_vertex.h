#pragma once
#include "ranally/ast/core/expression_vertex.h"
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

    friend class WhileVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   WhileVertex         (
                        std::shared_ptr<ExpressionVertex> const& condition,
                        StatementVertices const& true_statements,
                        StatementVertices const& false_statements);

                   ~WhileVertex        ()=default;

                   WhileVertex         (WhileVertex&&)=delete;

    WhileVertex&   operator=           (WhileVertex&&)=delete;

                   WhileVertex         (WhileVertex const&)=delete;

    WhileVertex&   operator=           (WhileVertex const&)=delete;

    std::shared_ptr<ExpressionVertex> const& condition() const;

    StatementVertices const& true_statements() const;

    StatementVertices& true_statements ();

    StatementVertices const& false_statements() const;

    StatementVertices& false_statements();

private:

    ExpressionVertexPtr _condition;

    StatementVertices _true_statements;

    StatementVertices _false_statements;

};

} // namespace ranally
