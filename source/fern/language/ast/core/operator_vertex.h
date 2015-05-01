// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/core/operation_vertex.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OperatorVertex:
    public OperationVertex
{

    friend class OperatorVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   OperatorVertex      (std::string const& name,
                                        ExpressionVertices const& expressions);

                   ~OperatorVertex     ()=default;

                   OperatorVertex      (OperatorVertex&&)=delete;

    OperatorVertex& operator=          (OperatorVertex&&)=delete;

                   OperatorVertex      (OperatorVertex const&)=delete;

    OperatorVertex& operator=          (OperatorVertex const&)=delete;

    std::string const&
                   symbol              () const;

private:

    std::string    _symbol;

};

} // namespace language
} // namespace fern
