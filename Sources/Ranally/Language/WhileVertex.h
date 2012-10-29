#pragma once
#include "Ranally/Language/ExpressionVertex.h"
#include "Ranally/Language/StatementVertex.h"


namespace ranally {
namespace language {

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
                        boost::shared_ptr<ExpressionVertex> const& condition,
                        StatementVertices const& trueStatements,
                        StatementVertices const& falseStatements);

                   ~WhileVertex        ();

  boost::shared_ptr<ExpressionVertex> const& condition() const;

  StatementVertices const& trueStatements() const;

  StatementVertices& trueStatements    ();

  StatementVertices const& falseStatements() const;

  StatementVertices& falseStatements   ();

private:

  ExpressionVertexPtr _condition;

  StatementVertices _trueStatements;

  StatementVertices _falseStatements;

};

} // namespace language
} // namespace ranally
