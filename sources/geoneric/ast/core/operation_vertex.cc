#include "geoneric/ast/core/operation_vertex.h"


namespace geoneric {

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

} // namespace geoneric
