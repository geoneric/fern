#include "StringVertex.h"



namespace ranally {

StringVertex::StringVertex(
  int lineNr,
  int colId,
  UnicodeString const& value)

  : ExpressionVertex(lineNr, colId),
    _value(value)

{
}



StringVertex::~StringVertex()
{
}



UnicodeString const& StringVertex::value() const
{
  return _value;
}

} // namespace ranally

