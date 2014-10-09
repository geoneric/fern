#pragma once
#include "fern/ast/core/expression_vertex.h"
#include "fern/ast/core/statement_vertex.h"


namespace fern {

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

                   ~AssignmentVertex   ()=default;

                   AssignmentVertex    (AssignmentVertex&&)=delete;

    AssignmentVertex& operator=        (AssignmentVertex&&)=delete;

                   AssignmentVertex    (AssignmentVertex const&)=delete;

    AssignmentVertex& operator=        (AssignmentVertex const&)=delete;

    ExpressionVertexPtr const& target  () const;

    ExpressionVertexPtr& target        ();

    void           set_expression      (ExpressionVertexPtr const& expression);

    ExpressionVertexPtr const& expression() const;

private:

    ExpressionVertexPtr _target;

    ExpressionVertexPtr _expression;

};

} // namespace fern
