#include "ScriptVertex.h"



namespace ranally {

ScriptVertex::ScriptVertex(
  StatementVertices const& statements)

  : _statements(statements)

{
}



ScriptVertex::~ScriptVertex()
{
}



StatementVertices const& ScriptVertex::statements() const
{
  return _statements;
}

} // namespace ranally

