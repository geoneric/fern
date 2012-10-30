#include "Ranally/Language/OperationVertex.h"


namespace ranally {
namespace language {

OperationVertex::OperationVertex(
    String const& name,
    ExpressionVertices const& expressions)

    : ExpressionVertex(name),
      _expressions(expressions)

{
}


OperationVertex::~OperationVertex()
{
}


ExpressionVertices const& OperationVertex::expressions() const
{
    return _expressions;
}


void OperationVertex::setOperation(
    operation::OperationPtr const& operation)
{
    assert(!_operation);
    _operation = operation;
}


operation::OperationPtr const& OperationVertex::operation() const
{
    return _operation;
}

} // namespace language
} // namespace ranally
