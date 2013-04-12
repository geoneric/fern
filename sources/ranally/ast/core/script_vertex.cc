#include "ranally/ast/core/script_vertex.h"
#include "ranally/ast/visitor/copy_visitor.h"


namespace ranally {

ScriptVertex::ScriptVertex(
    String const& source_name,
    StatementVertices const& statements)

    : SyntaxVertex(),
      _source_name(source_name),
      _statements(statements)

{
}


// ScriptVertex::ScriptVertex(
//   ScriptVertex const& other)
// 
//   : SyntaxVertex(other),
//     _source_name(other._source_name)
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


String const& ScriptVertex::source_name() const
{
    return _source_name;
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
