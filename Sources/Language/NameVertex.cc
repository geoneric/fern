#include "NameVertex.h"



namespace ranally {

NameVertex::NameVertex(
  UnicodeString const& name)

  : ExpressionVertex(name),
    _definition(0)

{
}



NameVertex::NameVertex(
  int lineNr,
  int colId,
  UnicodeString const& name)

  : ExpressionVertex(lineNr, colId, name),
    _definition(0)

{
}



NameVertex::~NameVertex()
{
}



void NameVertex::setDefinition(
  NameVertex* definition)
{
  assert(!_definition);
  assert(definition);
  _definition = definition;
}



NameVertex const* NameVertex::definition() const
{
  return _definition;
}

} // namespace ranally

