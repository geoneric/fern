#include "ranally/language/string_vertex.h"


namespace ranally {

StringVertex::StringVertex(
    int lineNr,
    int colId,
    String const& value)

    : ExpressionVertex(lineNr, colId, "\"" + value + "\""),
      _value(value)

{
}


String const& StringVertex::value() const
{
    return _value;
}

} // namespace ranally
