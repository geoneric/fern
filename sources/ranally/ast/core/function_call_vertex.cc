#include "ranally/ast/core/function_call_vertex.h"


namespace ranally {

FunctionCallVertex::FunctionCallVertex(
    String const& name,
    ExpressionVertices const& expressions)

    : OperationVertex(name, expressions)

{
}

} // namespace ranally
