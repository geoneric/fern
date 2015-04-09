// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/core/module_vertex.h"
// #include "fern/ast/visitor/copy_visitor.h"


namespace fern {

ModuleVertex::ModuleVertex(
    String const& source_name,
    std::shared_ptr<ScopeVertex> const& scope)

    : AstVertex(),
      _source_name(source_name),
      _scope(scope)

{
}


// ModuleVertex::ModuleVertex(
//   ModuleVertex const& other)
// 
//   : AstVertex(other),
//     _source_name(other._source_name)
// 
// {
//   BOOST_FOREACH(std::shared_ptr<StatementVertex> const& vertex,
//     _statements) {
//     CopyVisitor visitor;
//     vertex->Accept(visitor);
//     assert(visitor.statements().size() == 1);
//     _statements.emplace_back(visitor.statements()[0]);
//   }
// }


String const& ModuleVertex::source_name() const
{
    return _source_name;
}


std::shared_ptr<ScopeVertex> const& ModuleVertex::scope() const
{
    return _scope;
}


std::shared_ptr<ScopeVertex>& ModuleVertex::scope()
{
    return _scope;
}


// bool operator==(
//   ModuleVertex const& lhs,
//   ModuleVertex const& rhs)
// {
//   // EqualityVisitor visitor(&rhs);
//   // lhs.Accept(visitor);
//   // return visitor.equal();
//   return false;
// }



// bool operator!=(
//   ModuleVertex const& lhs,
//   ModuleVertex const& rhs)
// {
//   return !(lhs == rhs);
// }

} // namespace fern
