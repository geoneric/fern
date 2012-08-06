#include "Ranally/Language/ValidateVisitor.h"
#include <stdexcept>
#include "Ranally/Util/String.h"
#include "Ranally/Operation/Operation.h"
#include "Ranally/Language/FunctionVertex.h"



namespace ranally {
namespace language {

ValidateVisitor::ValidateVisitor()

  : Visitor()

{
}



ValidateVisitor::~ValidateVisitor()
{
}



void ValidateVisitor::Visit(
  FunctionVertex& vertex)
{
  // Find out if we know about an operation with the same name as this
  // function's name.
  if(!vertex.operation()) {
    throw std::runtime_error(("unknown function: " +
      vertex.name().encodeInUTF8()).c_str());
  }

  // ranally::operation::OperationPtr operation(_operations->operation(
  //   vertex.name()));

  // // Check if the number of arguments provided equals the required number of
  // // arguments.
  // // ...

  // // Check if the data type of each provided argument is accepted by the
  // // operation.
  // // ...

  // // Check if the value type of each provided argument is accepted by the
  // // operation.
  // // ...
}

} // namespace language
} // namespace ranally

