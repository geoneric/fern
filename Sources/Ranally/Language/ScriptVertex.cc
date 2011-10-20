#include "Ranally/Language/ScriptVertex.h"



namespace ranally {
namespace language {

ScriptVertex::ScriptVertex(
  UnicodeString const& sourceName,
  StatementVertices const& statements)

  : _sourceName(sourceName),
    _statements(statements)

{
}



ScriptVertex::~ScriptVertex()
{
}



UnicodeString const& ScriptVertex::sourceName() const
{
  return _sourceName;
}



StatementVertices const& ScriptVertex::statements() const
{
  return _statements;
}

} // namespace language
} // namespace ranally

