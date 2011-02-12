#include "Ranally/Language/SymbolTable.h"

#include <algorithm>
#include <cassert>
#include <boost/foreach.hpp>
#include "Ranally/Language/NameVertex.h"



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



//! Add a scope to the symbol table.
/*!
  \sa        popScope().

  All subsequent calls to addDefinition(NameVertex*) will add definitions
  to this new scope.
*/
void SymbolTable::pushScope()
{
  _scopes.push_back(Definitions());
}



//! Remove a scope from the symbol table.
/*!
  \warning   pushScope() must have been called first.
  \sa        pushScope().

  All definitions present in the current scope are removed.
*/
void SymbolTable::popScope()
{
  // For all definitions in the top-most scope, first remove them from the
  // definitions map. After that they can be removed from the scope stack.

  assert(!_scopes.empty());

  BOOST_FOREACH(NameVertex* definition, _scopes.back()) {
    Definitions& definitions(this->definitions(definition->name()));
    assert(std::find(definitions.begin(), definitions.end(), definition) !=
      definitions.end());
    definitions.remove(definition);

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



//! Return the scope level that contains the defintion of \a name.
/*!
  \param     name Name to look up scope level for.
  \return    Scope level.
*/
SymbolTable::size_type SymbolTable::scopeLevel(
  UnicodeString const& name) const
{
  assert(hasDefinition(name));

  // Iterate over each scope level until we find the level that contains the
  // requested definition.
  size_type result = _scopes.size();

  for(size_type i = 0; i < _scopes.size(); ++i) {
    size_type j = _scopes.size() - 1 - i;

    BOOST_FOREACH(NameVertex* definition, _scopes[j]) {
      if(definition->name() == name) {
        result = j + 1;
        break;
      }
    }
  }

  return result;
}



//! Add a definition to the current scope.
/*!
  \param     definition Definition to add.
  \warning   pushScope() must be called before definitions can be added to the
             symbol table.
*/
void SymbolTable::addDefinition(
  NameVertex* definition)
{
  // Add an empty list of definitions if a definition by this name does not
  // already exist.
  if(!hasDefinition(definition->name())) {
    _definitions[definition->name()] = Definitions();
  }

  // TODO If the name is already defined, we may want to issue a warning that
  //      - this new definition is hiding the existing one (nested scope), or
  //      - this new definition is overwriting the existing one (unnested
  //      scope).
  //      This warning should be conditional on some warning level.

  // TODO If the name is already defined in the current scope, this new
  //      definition should overwrite the previous one. Currently, the new
  //      definition is added to the collections.

  // Store the pointer in the list of definitions for this name. The
  // most recent definition is stored at the front of the list.
  Definitions& definitionsByName(definitions(definition->name()));
  definitionsByName.insert(definitionsByName.begin(), definition);

  // Add the pointer also to the list of definitions present in the current
  // scope. The most recent definition is stored at the front of the list.
  assert(!_scopes.empty());
  Definitions& definitionsByScope(_scopes.back());
  definitionsByScope.insert(definitionsByScope.begin(),
    definitionsByName.front());
}



bool SymbolTable::hasDefinition(
  UnicodeString const& name) const
{
  return _definitions.find(name) != _definitions.end();
}



NameVertex const* SymbolTable::definition(
  UnicodeString const& name) const
{
  assert(!definitions(name).empty());
  return definitions(name).front();
}



NameVertex* SymbolTable::definition(
  UnicodeString const& name)
{
  assert(!definitions(name).empty());
  return definitions(name).front();
}



bool SymbolTable::empty() const
{
  return size() == 0;
}



SymbolTable::size_type SymbolTable::size() const
{
  size_type result = 0;

  BOOST_FOREACH(Definitions const& definitions, _scopes) {
    result += definitions.size();
  }

  return result;
}

} // namespace language
} // namespace ranally

