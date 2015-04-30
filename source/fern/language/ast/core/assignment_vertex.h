// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/core/expression_vertex.h"
#include "fern/language/ast/core/statement_vertex.h"


namespace fern {
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

} // namespace language
} // namespace fern
