#include "Ranally/Language/StringVertex.h"



namespace ranally {
namespace language {

StringVertex::StringVertex(
  int lineNr,
  int colId,
  UnicodeString const& value)

  : ExpressionVertex(lineNr, colId, "\"" + value + "\""),
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

} // namespace language
} // namespace ranally
