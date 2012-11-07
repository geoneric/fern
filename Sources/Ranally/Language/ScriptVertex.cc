#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/CopyVisitor.h"


namespace ranally {

ScriptVertex::ScriptVertex(
    String const& sourceName,
    StatementVertices const& statements)

    : SyntaxVertex(),
      _sourceName(sourceName),
      _statements(statements)

{
}


// ScriptVertex::ScriptVertex(
//   ScriptVertex const& other)
// 
//   : SyntaxVertex(other),
//     _sourceName(other._sourceName)
// 
// {
//   BOOST_FOREACH(std::shared_ptr<StatementVertex> const& vertex,
//     _statements) {
//     CopyVisitor visitor;
//     vertex->Accept(visitor);
//     assert(visitor.statements().size() == 1);
//     _statements.push_back(visitor.statements()[0]);
//   }
// }


String const& ScriptVertex::sourceName() const
{
    return _sourceName;
}


StatementVertices const& ScriptVertex::statements() const
{
    return _statements;
}


StatementVertices& ScriptVertex::statements()
{
    return _statements;
}


// bool operator==(
//   ScriptVertex const& lhs,
//   ScriptVertex const& rhs)
// {
//   // EqualityVisitor visitor(&rhs);
//   // lhs.Accept(visitor);
//   // return visitor.equal();
//   return false;
// }



// bool operator!=(
//   ScriptVertex const& lhs,
//   ScriptVertex const& rhs)
// {
//   return !(lhs == rhs);
// }

} // namespace ranally
