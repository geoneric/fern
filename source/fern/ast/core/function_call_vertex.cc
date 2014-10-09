#include "fern/ast/core/function_call_vertex.h"


namespace fern {

FunctionCallVertex::FunctionCallVertex(
    String const& name,
    ExpressionVertices const& expressions)

    : OperationVertex(name, expressions)

{
}

} // namespace fern
