#include "Ranally/Language/OperationVertex.h"


namespace ranally {

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


void OperationVertex::setOperation(
    OperationPtr const& operation)
{
    assert(!_operation);
    _operation = operation;
}


OperationPtr const& OperationVertex::operation() const
{
    return _operation;
}

} // namespace ranally
