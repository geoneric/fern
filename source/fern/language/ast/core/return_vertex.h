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

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ReturnVertex:
    public StatementVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   ReturnVertex        ();

                   ReturnVertex        (ExpressionVertexPtr const& expression);

                   ~ReturnVertex       ()=default;

                   ReturnVertex        (ReturnVertex&&)=delete;

    ReturnVertex&  operator=           (ReturnVertex&&)=delete;

                   ReturnVertex        (ReturnVertex const&)=delete;

    ReturnVertex&  operator=           (ReturnVertex const&)=delete;

    ExpressionVertexPtr const& expression() const;

private:

    ExpressionVertexPtr _expression;

};

} // namespace fern
