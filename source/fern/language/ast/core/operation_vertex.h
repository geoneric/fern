// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/operation/core/operation.h"
#include "fern/language/ast/core/expression_vertex.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OperationVertex:
    public ExpressionVertex
{

    friend class OperationVertexTest;

public:

    virtual        ~OperationVertex    ()=default;

                   OperationVertex     (OperationVertex&&)=delete;

    OperationVertex& operator=         (OperationVertex&&)=delete;

                   OperationVertex     (OperationVertex const&)=delete;

    OperationVertex& operator=         (OperationVertex const&)=delete;

    ExpressionVertices const& expressions() const;

    void           set_operation       (OperationPtr const& operation);

    OperationPtr const& operation      () const;

protected:

                   OperationVertex     (std::string const& name,
                                        ExpressionVertices const& expressions);

private:

    ExpressionVertices _expressions;

    OperationPtr   _operation;

};

} // namespace language
} // namespace fern
