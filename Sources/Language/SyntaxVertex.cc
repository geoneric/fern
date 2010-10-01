#include "SyntaxVertex.h"



namespace ranally {

SyntaxVertex::SyntaxVertex(
  int lineNr,
  int colId)

  : _line(lineNr),
    _col(colId)

{
}



SyntaxVertex::~SyntaxVertex()
{
}



int SyntaxVertex::line() const
{
  return _line;
}



int SyntaxVertex::col() const
{
  return _col;
}

} // namespace ranally

