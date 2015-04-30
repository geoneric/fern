// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/operation_vertex.h"


namespace fern {

OperationVertex::OperationVertex(
    String const& name,
    ExpressionVertices const& expressions)

    : ExpressionVertex(name),
      _expressions(expressions)

{
}


ExpressionVertices const& OperationVertex::expressions() const
{
    return _expressions;
}


void OperationVertex::set_operation(
    OperationPtr const& operation)
{
    assert(!_operation);
    _operation = operation;
}


OperationPtr const& OperationVertex::operation() const
{
    return _operation;
}

} // namespace fern
