#include "Ranally/Language/ValidateVisitor.h"
#include "Ranally/Operation/Operation.h"
#include "Ranally/Language/FunctionVertex.h"



namespace ranally {
namespace language {

ValidateVisitor::ValidateVisitor(
  ranally::operation::OperationsPtr const& operations)

  : Visitor(),
    _operations(operations)

{
  assert(_operations);
}



ValidateVisitor::~ValidateVisitor()
{
}



void ValidateVisitor::Visit(
  FunctionVertex& vertex)
{
  // Find out if we know about an operation with the same name as this
  // function's name.
  if(!_operations->hasOperation(vertex.name())) {
    throw std::runtime_error(("unknown function: " +
      dev::encodeInUTF8(vertex.name())).c_str());
  }

  ranally::operation::OperationPtr operation(_operations->operation(
    vertex.name()));

  // Check if the number of arguments provided equals the required number of
  // arguments.
  // ...

  // Check if the data type of each provided argument is accepted by the
  // operation.
  // ...

  // Check if the value type of each provided argument is accepted by the
  // operation.
  // ...


}

} // namespace language
} // namespace ranally

