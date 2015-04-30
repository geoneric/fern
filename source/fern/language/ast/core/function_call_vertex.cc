// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/function_call_vertex.h"


namespace fern {
namespace language {

FunctionCallVertex::FunctionCallVertex(
    String const& name,
    ExpressionVertices const& expressions)

    : OperationVertex(name, expressions)

{
}

} // namespace language
} // namespace fern
