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

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FunctionCallVertex:
    public OperationVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   FunctionCallVertex  (String const& name,
                                        ExpressionVertices const& expressions);

                   ~FunctionCallVertex ()=default;

                   FunctionCallVertex  (FunctionCallVertex&&)=delete;

    FunctionCallVertex& operator=      (FunctionCallVertex&&)=delete;

                   FunctionCallVertex  (FunctionCallVertex const&)=delete;

    FunctionCallVertex& operator=      (FunctionCallVertex const&)=delete;

private:

};

} // namespace fern
