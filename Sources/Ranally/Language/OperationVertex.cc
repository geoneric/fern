#include "Ranally/Language/OperationVertex.h"



namespace ranally {
namespace language {

OperationVertex::OperationVertex(
  UnicodeString const& name)

  : ExpressionVertex(name)

{
}



OperationVertex::~OperationVertex()
{
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

