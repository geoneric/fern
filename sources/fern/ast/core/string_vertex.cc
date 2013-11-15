#include "fern/ast/core/string_vertex.h"


namespace fern {

StringVertex::StringVertex(
    int line_nr,
    int col_id,
    String const& value)

    : ExpressionVertex(line_nr, col_id, "\"" + value + "\""),
      _value(value)

{
}


String const& StringVertex::value() const
{
    return _value;
}

} // namespace fern
