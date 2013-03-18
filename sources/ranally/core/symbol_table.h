#pragma once
#include <cassert>
#include <boost/range/adaptor/reversed.hpp>
#include "ranally/core/scope.h"
#include "ranally/core/string.h"


namespace ranally {

//! Datastructure for keeping track of symbols and their value.
/*!
  The table is able to store multiple values of the same name and supports
  scoping.

  Values are added to the current scope using add_symbol. Make sure
  that such a scope exists. After creation of a SymbolTable instance,
  push_scope() must be called before symbols can be added. You can make
  multiple calls to push_scope() in case of nested scopes. When filling the
  table, make sure to match each call to push_scope() with a call to
  pop_scope().
*/
template<
    class T>
class SymbolTable
{

public:

    //! Type for scope levels.
    typedef typename std::vector<Scope<T>>::size_type size_type;

    //! Construct an empty symbol table.
    /*!
      \warning   Call push_scope before adding values.
    */
                   SymbolTable         ()=default;

                   ~SymbolTable        ()=default;

                   SymbolTable         (SymbolTable&&)=delete;

    SymbolTable&   operator=           (SymbolTable&&)=delete;

                   SymbolTable         (SymbolTable const&)=delete;

    SymbolTable&   operator=           (SymbolTable const&)=delete;

    void           push_scope          ();

    void           pop_scope           ();

    size_type      scope_level         () const;

    size_type      scope_level         (String const& name) const;

    void           add_value           (String const& name,
                                        T const& value);

    bool           has_value           (String const& name) const;

    T              value               (String const& name) const;

    bool           empty               () const;

    size_type      size                () const;

private:

    //! Values by scope level.
    std::vector<Scope<T>> _scopes;

};


//! Add a scope to the symbol table.
/*!
  \sa        pop_scope().

  All subsequent calls to add_value(String const&, T const&) will add
  values to this new scope.
*/
template<
    class T>
void SymbolTable<T>::push_scope()
{
    _scopes.push_back(Scope<T>());
}


//! Remove a scope from the symbol table.
/*!
  \warning   push_scope() must have been called first.
  \sa        push_scope().

  All values present in the current scope are removed.
*/
template<
    class T>
void SymbolTable<T>::pop_scope()
{
    assert(!_scopes.empty());
    _scopes.pop_back();
}


//! Return the current scope level.
/*!
  \return    The 0-based scope level.

  Normally, this function will return a value larger than zero. Only when no
  scopes are pushed (using push_scope()) will this function return zero.
  Global scope is at scope level 1.
*/
template<
    class T>
typename SymbolTable<T>::size_type SymbolTable<T>::scope_level() const
{
    return _scopes.size();
}


//! Return the scope level that contains the defintion of \a name.
/*!
  \param     name Name to look up scope level for.
  \return    Scope level.
*/
template<
    class T>
typename SymbolTable<T>::size_type SymbolTable<T>::scope_level(
    String const& name) const
{
    assert(has_value(name));

    // Iterate over each scope level until we find the level that contains the
    // requested value.
    size_type result = _scopes.size();

    if(!_scopes.empty()) {
        do {
            --result;

            if(_scopes[result].has_value(name)) {
                result += 1;
                break;
            }
        } while(result != 0);
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
template<
    class T>
void SymbolTable<T>::add_value(
    String const& name,
    T const& value)
{
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

    _scopes.back().set_value(name, value);
}


template<
    class T>
bool SymbolTable<T>::has_value(
    String const& name) const
{
    bool result = false;

    for(auto scope: _scopes) {
        if(scope.has_value(name)) {
            result = true;
            break;
        }
    }

    return result;
}


template<
    class T>
T SymbolTable<T>::value(
    String const& name) const
{
    assert(has_value(name));
    T result;

    for(auto scope: _scopes | boost::adaptors::reversed) {
        if(scope.has_value(name)) {
            result = scope.value(name);
            break;
        }
    }

    return result;
}


template<
    class T>
bool SymbolTable<T>::empty() const
{
    return size() == 0;
}


template<
    class T>
typename SymbolTable<T>::size_type SymbolTable<T>::size() const
{
    size_type result = 0;

    for(auto scope: _scopes) {
        result += scope.size();
    }

    return result;
}

} // namespace ranally
