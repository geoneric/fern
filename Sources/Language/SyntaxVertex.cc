#include "SyntaxVertex.h"



namespace ranally {

SyntaxVertex::SyntaxVertex()

  : _successor(0)

{
}



SyntaxVertex::SyntaxVertex(
  int lineNr,
  int colId)

  : _line(lineNr),
    _col(colId),
    _successor(0)

{
}



SyntaxVertex::~SyntaxVertex()
{
}



void SyntaxVertex::setPosition(
  int lineNr,
  int colId)
{
  _line = lineNr;
  _col = colId;
}



int SyntaxVertex::line() const
{
  return _line;
}



int SyntaxVertex::col() const
{
  return _col;
}



SyntaxVertex* SyntaxVertex::successor()
{
  assert(_successor);
  return _successor;
}



SyntaxVertex const* SyntaxVertex::successor() const
{
  assert(_successor);
  return _successor;
}



void SyntaxVertex::setSuccessor(
  SyntaxVertex* successor)
{
  assert(!_successor);
  assert(successor);
  _successor = successor;
  assert(_successor);
}


} // namespace ranally

