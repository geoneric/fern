#include "Ranally/Language/ScriptVertex.h"



namespace ranally {
namespace language {

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

} // namespace language
} // namespace ranally

