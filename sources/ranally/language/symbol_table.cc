#include "ranally/language/symbol_table.h"
#include <algorithm>
#include <cassert>
#include "ranally/language/name_vertex.h"


namespace ranally {

//! Destruct a symbol table.
/*!
  All values in all scopes are removed.
*/
SymbolTable::~SymbolTable()
{
    while(!_scopes.empty()) {
        pop_scope();
    }

    assert(_scopes.empty());
    assert(_values.empty());
}


// SymbolTable::Values const& SymbolTable::values(
//     String const& name) const
// {
//     assert(has_value(name));
//     return _values.find(name)->second;
// }
// 
// 
// SymbolTable::Values& SymbolTable::values(
//     String const& name)
// {
//     assert(has_value(name));
//     return _values.find(name)->second;
// }


//! Add a scope to the symbol table.
/*!
  \sa        pop_scope().

  All subsequent calls to add_value(String const&, NameVertex*) will add
  values to this new scope.
*/
void SymbolTable::push_scope()
{
    _scopes.push_back(Values());
    _scopes2.push_back(Scope<SymbolTable::T>());
}


//! Remove a scope from the symbol table.
/*!
  \warning   push_scope() must have been called first.
  \sa        push_scope().

  All values present in the current scope are removed.
*/
void SymbolTable::pop_scope()
{
    // For all values in the top-most scope, first remove them from the
    // values map. After that they can be removed from the scope stack.

    assert(!_scopes.empty());

    for(auto value: _scopes.back()) {
        // TODO...
        Values& values(this->values(value->name()));
        assert(std::find(values.begin(), values.end(), value) !=
            values.end());
        values.remove(value);

        // Erase the list of values for the current name if the list is
        // empty.
        if(values.empty()) {
            // TODO...
            _values.erase(value->name());
        }
    }

    _scopes.pop_back();
    _scopes2.pop_back();
}


//! Return the current scope level.
/*!
  \return    The 0-based scope level.

  Normally, this function will return a value larger than zero. Only when no
  scopes are pushed (using push_scope()) will this function return zero.
*/
SymbolTable::size_type SymbolTable::scope_level() const
{
    // return _scopes.size();
    return _scopes2.size();
}


//! Return the scope level that contains the defintion of \a name.
/*!
  \param     name Name to look up scope level for.
  \return    Scope level.
*/
SymbolTable::size_type SymbolTable::scope_level(
    String const& name) const
{
    assert(has_value(name));

    // Iterate over each scope level until we find the level that contains the
    // requested value.
    size_type result = _scopes.size();

    for(size_type i = 0; i < _scopes.size(); ++i) {
        size_type j = _scopes.size() - 1 - i;

        for(auto value: _scopes[j]) {
            // TODO...
            if(value->name() == name) {
                result = j + 1;
                break;
            }
        }
    }

    return result;
}


//! Add a value to the current scope.
/*!
  \param     name Name of value to add.
  \param     value Value to add.
  \warning   push_scope() must be called before values can be added to the
             symbol table.
  \todo      If name is already defined in an outer scope, add it to that one.
*/
void SymbolTable::add_value(
    String const& name,
    T const& value)
{
    // Add an empty list of values if a value by this name does not
    // already exist.
    if(!has_value(name)) {
        _values[name] = Values();
    }

    // TODO If the name is already defined, we may want to issue a warning that
    //      - this new value is hiding the existing one (nested scope), or
    //      - this new value is overwriting the existing one (unnested
    //      scope).
    //      This warning should be conditional on some warning level.

    // TODO If the name is already defined in the current scope, this new
    //      value should overwrite the previous one. Currently, the new
    //      value is added to the collections.

    // TODO In some cases, the new value should overwrite the previous one,
    //      but in other cases, the new value should just be added to the
    //      list of value. This happens when a global identifier is assigned
    //      to in an if/while script, for example. The value of the identifier
    //      may be updated, but this depends on the runtime condition. So, if
    //      multiple definitions with the same name exist in the same scope,
    //      then they should be treated as possible definitions. They should be
    //      checked for compatibility (same/convertable types).

    // Store the pointer in the list of definitions for this name. The
    // most recent value is stored at the front of the list.
    Values& definitions_by_name(values(name));
    definitions_by_name.insert(definitions_by_name.begin(), value);

    // Add the pointer also to the list of values present in the current
    // scope. The most recent value is stored at the front of the list.
    assert(!_scopes.empty());
    Values& definitions_by_scope(_scopes.back());
    definitions_by_scope.insert(definitions_by_scope.begin(),
        definitions_by_name.front());
}


bool SymbolTable::has_value(
    String const& name) const
{
    return _values.find(name) != _values.end();
}


SymbolTable::T const& SymbolTable::value(
    String const& name) const
{
    assert(!values(name).empty());
    return values(name).front();
}


bool SymbolTable::empty() const
{
    return size() == 0;
}


SymbolTable::size_type SymbolTable::size() const
{
    size_type result = 0;

    for(auto values: _scopes) {
        result += values.size();
    }

    return result;
}

} // namespace ranally
