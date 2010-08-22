#include "SyntaxVertex.h"



namespace ranally {

SyntaxVertex::SyntaxVertex(
  int lineNr,
  int colId,
  UnicodeString const& value)

  : _line(lineNr),
    _col(colId),
    _value(value)

{
}



SyntaxVertex::~SyntaxVertex()
{
}

} // namespace ranally

