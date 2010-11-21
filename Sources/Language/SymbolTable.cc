#include "SymbolTable.h"

#include <algorithm>
#include <cassert>
#include <boost/foreach.hpp>
#include "Definition.h"



namespace ranally {

SymbolTable::SymbolTable()
{
}



SymbolTable::~SymbolTable()
{
}



SymbolTable::Definitions const& SymbolTable::definitions(
  UnicodeString const& name) const
{
  assert(_definitions.find(name) != _definitions.end());
  return _definitions.find(name)->second;
}



SymbolTable::Definitions& SymbolTable::definitions(
  UnicodeString const& name)
{
  assert(_definitions.find(name) != _definitions.end());
  return _definitions.find(name)->second;
}



void SymbolTable::pushScope()
{
  _scopes.push(Definitions());
}




void SymbolTable::popScope()
{
  // For all definitions in the top-most scope, first remove them from the
  // definitions map. After that they can be removed from the scope stack.

  assert(!_scopes.empty());

  BOOST_FOREACH(Definition* definition, _scopes.top()) {
    Definitions& definitions(this->definitions(definition->name()));
    assert(std::find(definitions.begin(), definitions.end(), definition) !=
      definitions.end());
    definitions.remove(definition);
    delete definition;
  }

  _scopes.pop();
}




void SymbolTable::addDefinition(
  Definition* definition)
{
  if(_definitions.find(definition->name()) == _definitions.end()) {
    _definitions[definition->name()] = Definitions();
  }

  {
    Definitions& definitions(this->definitions(definition->name()));
    definitions.insert(definitions.begin(), definition);
  }

  assert(!_scopes.empty());

  {
    Definitions& definitions(_scopes.top());
    definitions.insert(definitions.begin(), definition);
  }
}



Definition const& SymbolTable::definition(
  UnicodeString const& name) const
{
  assert(!definitions(name).empty());
  return *definitions(name).front();
}



Definition& SymbolTable::definition(
  UnicodeString const& name)
{
  assert(!definitions(name).empty());
  return *definitions(name).front();
}

} // namespace ranally

