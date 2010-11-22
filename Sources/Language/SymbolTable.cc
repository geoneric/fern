#include "SymbolTable.h"

#include <algorithm>
#include <cassert>
#include <boost/foreach.hpp>
#include "Definition.h"



namespace ranally {
namespace language {

//! Construct an empty symbol table.
/*!
  \warning   Call pushScope before adding definitions.
*/
SymbolTable::SymbolTable()
{
}



//! Destruct a symbol table.
/*!
  All definitions in all scopes are removed.
*/
SymbolTable::~SymbolTable()
{
  while(!_scopes.empty()) {
    popScope();
  }

  assert(_scopes.empty());
  assert(_definitions.empty());
}



SymbolTable::Definitions const& SymbolTable::definitions(
  UnicodeString const& name) const
{
  assert(hasDefinition(name));
  return _definitions.find(name)->second;
}



SymbolTable::Definitions& SymbolTable::definitions(
  UnicodeString const& name)
{
  assert(hasDefinition(name));
  return _definitions.find(name)->second;
}



void SymbolTable::pushScope()
{
  _scopes.push_back(Definitions());
}




void SymbolTable::popScope()
{
  // For all definitions in the top-most scope, first remove them from the
  // definitions map. After that they can be removed from the scope stack.

  assert(!_scopes.empty());

  BOOST_FOREACH(Definition* definition, _scopes.front()) {
    Definitions& definitions(this->definitions(definition->name()));
    assert(std::find(definitions.begin(), definitions.end(), definition) !=
      definitions.end());
    definitions.remove(definition);
    delete definition;

    // Erase the list of definitions for the current name if the list is empty.
    if(definitions.empty()) {
      _definitions.erase(definition->name());
    }
  }

  _scopes.pop_back();
}



//! Return the current scope level.
/*!
  \return    The 0-based scope level.

  Normally, this function will return a value larger than zero. Only when no
  scopes are pushed (using pushScope()) will this function return zero.
*/
SymbolTable::size_type SymbolTable::scopeLevel() const
{
  return _scopes.size();
}



SymbolTable::size_type SymbolTable::scopeLevel(
  UnicodeString const& name) const
{
  assert(hasDefinition(name));

  // Iterate over each scope level untill we find the level that contains the
  // requested definition.
  size_type result = _scopes.size();

  for(size_type i = 0; i < _scopes.size(); ++i) {
    size_type j = _scopes.size() - 1 - i;

    BOOST_FOREACH(Definition* definition, _scopes[j]) {
      if(definition->name() == name) {
        result = j + 1;
        break;
      }
    }
  }

  return result;
}



//! Adds a definition to the current scope.
/*!
  \param     definition Definition to add.
  \warning   pushScope() must be called before definitions can be added to the
             symbol table.
  \sa        .
*/
void SymbolTable::addDefinition(
  Definition const& definition)
{
  // Add an empty list of definitions if a definition by this name does not
  // already exist.
  if(!hasDefinition(definition.name())) {
    _definitions[definition.name()] = Definitions();
  }

  // TODO If the name is already defined, we may want to issue a warning that
  //      - this new definition is hiding the existing one (nested scope), or
  //      - this new definition is overwriting the existing one (unnested
  //      scope).
  //      This warning should be conditional on some warning level.

  // Create a copy on the stack and store the pointer in the list of
  // definitions for this name.
  Definitions& definitionsByName(definitions(definition.name()));
  definitionsByName.insert(definitionsByName.begin(),
    new Definition(definition));

  // Add the pointer also to the list of definitions present in the current
  // scope.
  assert(!_scopes.empty());
  Definitions& definitionsByScope(_scopes.front());
  definitionsByScope.insert(definitionsByScope.begin(),
    definitionsByName.front());
}



bool SymbolTable::hasDefinition(
  UnicodeString const& name) const
{
  return _definitions.find(name) != _definitions.end();
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

} // namespace language
} // namespace ranally

