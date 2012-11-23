#pragma once
#include "ranally/language/expression_vertex.h"
#include "ranally/language/statement_vertex.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  Threading an if-statement results in multiple successor vertices being stored in the
  base-class' successor collection.
  - The first successor always points to the first vertex of the true-block's
    control flow graph.
  - If there is a false-block, then the second successor points to the first vertex of
    its control flow graph.
  - The last successor (the second or third, depending on whether or not a false-block
    is present) points to the successor of the whole statement.

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
                        StatementVertices const& trueStatements,
                        StatementVertices const& falseStatements);

                   ~IfVertex           ()=default;

                   IfVertex            (IfVertex&&)=delete;

    IfVertex&      operator=           (IfVertex&&)=delete;

                   IfVertex            (IfVertex const&)=delete;

    IfVertex&      operator=           (IfVertex const&)=delete;

    std::shared_ptr<ExpressionVertex> const& condition() const;

    StatementVertices const& trueStatements() const;

    StatementVertices& trueStatements  ();

    StatementVertices const& falseStatements() const;

    StatementVertices& falseStatements ();

private:

    ExpressionVertexPtr _condition;

    StatementVertices _trueStatements;

    StatementVertices _falseStatements;

};

} // namespace ranally
