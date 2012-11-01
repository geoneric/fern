#include "Ranally/Language/StringVertex.h"


namespace ranally {

StringVertex::StringVertex(
    int lineNr,
    int colId,
    String const& value)

    : language::ExpressionVertex(lineNr, colId, "\"" + value + "\""),
      _value(value)

{
}


StringVertex::~StringVertex()
{
}


String const& StringVertex::value() const
{
    return _value;
}

} // namespace ranally
