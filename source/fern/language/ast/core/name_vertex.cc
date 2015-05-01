// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/ast/core/name_vertex.h"
#include <cassert>


namespace fern {
namespace language {

NameVertex::NameVertex(
    std::string const& name)

    : ExpressionVertex(name) // ,
      // _definition(0)

{
}


NameVertex::NameVertex(
    int line_nr,
    int col_id,
    std::string const& name)

    : ExpressionVertex(line_nr, col_id, name) // ,
      // _definition(0)

{
}


// void NameVertex::set_definition(
//   NameVertex* definition)
// {
//   assert(!_definition);
//   assert(definition);
//   _definition = definition;
// }
// 
// 
// 
// NameVertex const* NameVertex::definition() const
// {
//   return _definition;
// }



// NameVertex* NameVertex::definition()
// {
//   return _definition;
// }


void NameVertex::add_definition(
    NameVertex* vertex)
{
    _definitions.emplace_back(vertex);
}


std::vector<NameVertex*> const& NameVertex::definitions() const
{
    return _definitions;
}


void NameVertex::add_use(
    NameVertex* vertex)
{
    // Either the definition vertex is not set yet, or it is equal to this.
    // assert(!_definition || _definition == this);
    assert(_definitions.empty() ||
        (_definitions.size() == 1 && _definitions[0] == this));
    assert(vertex);
    assert(vertex != this);
    assert(name() == vertex->name());
    _uses.emplace_back(vertex);
}


//! Return the collection of vertices that represent uses of this name.
/*!
  \return    Collection of vertices.
  \warning   The collection is only relevant if this vertex represents the
             definition.
*/
std::vector<NameVertex*> const& NameVertex::uses() const
{
    return _uses;
}

} // namespace language
} // namespace fern
