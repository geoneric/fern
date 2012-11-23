#pragma once
#include "Ranally/Language/expression_vertex.h"
#include "Ranally/Language/statement_vertex.h"


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
                        StatementVertices const& trueStatements,
                        StatementVertices const& falseStatements);

                   ~WhileVertex        ()=default;

                   WhileVertex         (WhileVertex&&)=delete;

    WhileVertex&   operator=           (WhileVertex&&)=delete;

                   WhileVertex         (WhileVertex const&)=delete;

    WhileVertex&   operator=           (WhileVertex const&)=delete;

    std::shared_ptr<ExpressionVertex> const& condition() const;

    StatementVertices const& trueStatements() const;

    StatementVertices& trueStatements    ();

    StatementVertices const& falseStatements() const;

    StatementVertices& falseStatements   ();

private:

    ExpressionVertexPtr _condition;

    StatementVertices _trueStatements;

    StatementVertices _falseStatements;

};

} // namespace ranally
