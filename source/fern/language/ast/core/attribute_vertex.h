// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/core/name_vertex.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AttributeVertex:
    public ExpressionVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   AttributeVertex     (
                        std::shared_ptr<ExpressionVertex> const& expression,
                        std::string const& member_name);

                   ~AttributeVertex    ()=default;

                   AttributeVertex     (AttributeVertex&&)=delete;

    AttributeVertex&
                   operator=           (AttributeVertex&&)=delete;

                   AttributeVertex     (AttributeVertex const&)=delete;

    AttributeVertex&
                   operator=         (AttributeVertex const&)=delete;

    std::string const&
                   symbol              () const;

    std::shared_ptr<ExpressionVertex> const& expression() const;

    std::string const&
                   member_name         () const;

private:

    std::string const _symbol;

    ExpressionVertexPtr _expression;

    std::string    _member_name;

};

} // namespace language
} // namespace fern
