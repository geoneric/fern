#pragma once
#include "ranally/language/expression_vertex.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SubscriptVertex:
    public ExpressionVertex
{

    friend class SubscriptVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   SubscriptVertex     (
                        std::shared_ptr<ExpressionVertex> const& expression,
                        std::shared_ptr<ExpressionVertex> const& selection);

                   ~SubscriptVertex    ()=default;

                   SubscriptVertex     (SubscriptVertex&&)=delete;

    SubscriptVertex&      operator=    (SubscriptVertex&&)=delete;

                   SubscriptVertex     (SubscriptVertex const&)=delete;

    SubscriptVertex&      operator=    (SubscriptVertex const&)=delete;

    std::shared_ptr<ExpressionVertex> const& expression() const;

    std::shared_ptr<ExpressionVertex> const& selection() const;

private:

    ExpressionVertexPtr _expression;

    ExpressionVertexPtr _selection;

};

} // namespace ranally
