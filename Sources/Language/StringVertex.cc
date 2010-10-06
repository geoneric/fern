#include "StringVertex.h"



namespace ranally {

StringVertex::StringVertex(
  int lineNr,
  int colId,
  UnicodeString const& string)

  : ExpressionVertex(lineNr, colId),
    _string(string)

{
}



StringVertex::~StringVertex()
{
}

} // namespace ranally

