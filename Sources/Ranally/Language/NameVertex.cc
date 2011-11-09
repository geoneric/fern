#include "Ranally/Language/NameVertex.h"



namespace ranally {
namespace language {

NameVertex::NameVertex(
  UnicodeString const& name)

  : ExpressionVertex(name) // ,
    // _definition(0)

{
}



NameVertex::NameVertex(
  int lineNr,
  int colId,
  UnicodeString const& name)

  : ExpressionVertex(lineNr, colId, name) // ,
    // _definition(0)

{
}



NameVertex::~NameVertex()
{
}



// void NameVertex::setDefinition(
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



void NameVertex::addDefinition(
  NameVertex* vertex)
{
  _definitions.push_back(vertex);
}



std::vector<NameVertex*> const& NameVertex::definitions() const
{
  return _definitions;
}



void NameVertex::addUse(
  NameVertex* vertex)
{
  // Either the definition vertex is not set yet, or it is equal to this.
  // assert(!_definition || _definition == this);
  assert(_definitions.empty() ||
    (_definitions.size() == 1 && _definitions[0] == this));
  assert(vertex);
  assert(vertex != this);
  assert(name() == vertex->name());
  _uses.push_back(vertex);
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
} // namespace ranally

