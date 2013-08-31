#pragma once
#include "ranally/ast/core/expression_vertex.h"


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

    String const&  symbol              () const;

    std::shared_ptr<ExpressionVertex> const& expression() const;

    std::shared_ptr<ExpressionVertex> const& selection() const;

private:

    String const   _symbol;

    //! Expression being subscripted.
    ExpressionVertexPtr _expression;

    //! Expression that selects from the expression being subscripted.
    ExpressionVertexPtr _selection;

};

} // namespace ranally
