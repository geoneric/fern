#include "ranally/language/symbol_table.h"
#include <algorithm>
#include <cassert>
#include "ranally/language/name_vertex.h"


namespace ranally {

//! Destruct a symbol table.
/*!
  All definitions in all scopes are removed.
*/
SymbolTable::~SymbolTable()
{
    while(!_scopes.empty()) {
        pop_scope();
    }

    assert(_scopes.empty());
    assert(_definitions.empty());
}


SymbolTable::Definitions const& SymbolTable::definitions(
    String const& name) const
{
    assert(has_definition(name));
    return _definitions.find(name)->second;
}


SymbolTable::Definitions& SymbolTable::definitions(
    String const& name)
{
    assert(has_definition(name));
    return _definitions.find(name)->second;
}


//! Add a scope to the symbol table.
/*!
  \sa        pop_scope().

  All subsequent calls to add_definition(NameVertex*) will add definitions
  to this new scope.
*/
void SymbolTable::push_scope()
{
    _scopes.push_back(Definitions());
}


//! Remove a scope from the symbol table.
/*!
  \warning   push_scope() must have been called first.
  \sa        push_scope().

  All definitions present in the current scope are removed.
*/
void SymbolTable::pop_scope()
{
    // For all definitions in the top-most scope, first remove them from the
    // definitions map. After that they can be removed from the scope stack.

    assert(!_scopes.empty());

    for(auto definition: _scopes.back()) {
        Definitions& definitions(this->definitions(definition->name()));
        assert(std::find(definitions.begin(), definitions.end(), definition) !=
            definitions.end());
        definitions.remove(definition);

        // Erase the list of definitions for the current name if the list is
        // empty.
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
  scopes are pushed (using push_scope()) will this function return zero.
*/
SymbolTable::size_type SymbolTable::scope_level() const
{
    return _scopes.size();
}


//! Return the scope level that contains the defintion of \a name.
/*!
  \param     name Name to look up scope level for.
  \return    Scope level.
*/
SymbolTable::size_type SymbolTable::scope_level(
    String const& name) const
{
    assert(has_definition(name));

    // Iterate over each scope level until we find the level that contains the
    // requested definition.
    size_type result = _scopes.size();

    for(size_type i = 0; i < _scopes.size(); ++i) {
        size_type j = _scopes.size() - 1 - i;

        for(auto definition: _scopes[j]) {
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
  \warning   push_scope() must be called before definitions can be added to the
             symbol table.
  \todo      If name is already defined in an outer scope, add it to that one.
*/
void SymbolTable::add_definition(
    NameVertex* definition)
{
    // Add an empty list of definitions if a definition by this name does not
    // already exist.
    if(!has_definition(definition->name())) {
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

    // TODO In some cases, the new definition should overwrite the previous one,
    //      but in other cases, the new definition should just be added to the
    //      list of definition. This happens when a global identifier is assigned
    //      to in an if/while script, for example. The value of the identifier
    //      may be updated, but this depends on the runtime condition. So, if
    //      multiple definitions with the same name exist in the same scope,
    //      then they should be treated as possible definitions. They should be
    //      checked for compatibility (same/convertable types).

    // Store the pointer in the list of definitions for this name. The
    // most recent definition is stored at the front of the list.
    Definitions& definitions_by_name(definitions(definition->name()));
    definitions_by_name.insert(definitions_by_name.begin(), definition);

    // Add the pointer also to the list of definitions present in the current
    // scope. The most recent definition is stored at the front of the list.
    assert(!_scopes.empty());
    Definitions& definitions_by_scope(_scopes.back());
    definitions_by_scope.insert(definitions_by_scope.begin(),
        definitions_by_name.front());
}


bool SymbolTable::has_definition(
    String const& name) const
{
    return _definitions.find(name) != _definitions.end();
}


NameVertex const* SymbolTable::definition(
    String const& name) const
{
    assert(!definitions(name).empty());
    return definitions(name).front();
}


NameVertex* SymbolTable::definition(
    String const& name)
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

    for(auto definitions: _scopes) {
        result += definitions.size();
    }

    return result;
}

} // namespace ranally
