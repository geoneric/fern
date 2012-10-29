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
class AssignmentVertex:
  public StatementVertex
{

  friend class AssignmentVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

                   AssignmentVertex    (ExpressionVertexPtr const& target,
                                        ExpressionVertexPtr const& expression);

                   ~AssignmentVertex   ();

  ExpressionVertexPtr const& target    () const;

  ExpressionVertexPtr& target          ();

  void             setExpression       (ExpressionVertexPtr const& expression);

  ExpressionVertexPtr const& expression() const;

private:

  ExpressionVertexPtr _target;

  ExpressionVertexPtr _expression;

};

} // namespace language
} // namespace ranally
