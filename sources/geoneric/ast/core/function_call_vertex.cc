#include "geoneric/ast/core/function_call_vertex.h"


namespace geoneric {

FunctionCallVertex::FunctionCallVertex(
    String const& name,
    ExpressionVertices const& expressions)

    : OperationVertex(name, expressions)

{
}

} // namespace geoneric
