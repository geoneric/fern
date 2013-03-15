#pragma once
#include <list>
#include <map>
#include <vector>
#include "ranally/language/scope.h"
#include "ranally/core/string.h"


namespace ranally {

class NameVertex;

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
class SymbolTable
{

    friend class SymbolTableTest;

public:

    typedef NameVertex* T;

    //! Type for lists of values.
    typedef std::list<T> Values;

    //! Type for scope levels.
    typedef std::vector<Values>::size_type size_type;

    //! Construct an empty symbol table.
    /*!
      \warning   Call push_scope before adding values.
    */
                   SymbolTable         ()=default;

                   ~SymbolTable        ();

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

    T const&       value               (String const& name) const;

    bool           empty               () const;

    size_type      size                () const;

private:

    //! Values by name.
    std::map<String, Values> _values;

    //! Values by scope level.
    std::vector<Values> _scopes;

    //! Values by scope level.
    std::vector<Scope<T>> _scopes2;

    // Values const&  values              (String const& name) const;

    // Values&        values              (String const& name);

};

} // namespace ranally
