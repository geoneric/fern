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


namespace fern {
namespace language {

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

    std::string const&
                   symbol              () const;

    std::shared_ptr<ExpressionVertex> const& expression() const;

    std::shared_ptr<ExpressionVertex> const& selection() const;

private:

    std::string const _symbol;

    //! Expression being subscripted.
    ExpressionVertexPtr _expression;

    //! Expression that selects from the expression being subscripted.
    ExpressionVertexPtr _selection;

};

} // namespace language
} // namespace fern
