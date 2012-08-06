#include "Ranally/Language/FunctionVertex.h"



namespace ranally {
namespace language {

FunctionVertex::FunctionVertex(
  String const& name,
  ExpressionVertices const& expressions)

  : OperationVertex(name, expressions)

{
}



FunctionVertex::~FunctionVertex()
{
}

} // namespace language
} // namespace ranally

