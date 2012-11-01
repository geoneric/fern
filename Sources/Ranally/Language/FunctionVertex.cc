#include "Ranally/Language/FunctionVertex.h"


namespace ranally {

FunctionVertex::FunctionVertex(
    String const& name,
    language::ExpressionVertices const& expressions)

    : OperationVertex(name, expressions)

{
}


FunctionVertex::~FunctionVertex()
{
}

} // namespace ranally
