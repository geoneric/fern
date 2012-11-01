#include "Ranally/Language/OperationVertex.h"


namespace ranally {

OperationVertex::OperationVertex(
    String const& name,
    language::ExpressionVertices const& expressions)

    : language::ExpressionVertex(name),
      _expressions(expressions)

{
}


OperationVertex::~OperationVertex()
{
}


language::ExpressionVertices const& OperationVertex::expressions() const
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

} // namespace ranally
