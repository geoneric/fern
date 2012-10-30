#pragma once
#include <list>
#include <map>
#include <vector>
#include "Ranally/Util/String.h"


namespace ranally {
namespace language {

class NameVertex;

//! Datastructure for keeping track of definitions.
/*!
  The table is able to store multiple definitions of the same name and supports
  scoping.

  Definitions are added to the current scope using addDefinition. Make sure
  that such a scope exists. After creation of a SymbolTable instance,
  pushScope() must be called before identifiers can be added. You can make
  multiple calls to pushScope() in case of nested scopes. When filling the
  table, make sure to match each call to pushScope() with a call to popScope().
*/
class SymbolTable
{

    friend class SymbolTableTest;

public:

    //! Type for lists of definitions.
    typedef std::list<NameVertex*> Definitions;

    //! Type for scope levels.
    typedef std::vector<Definitions>::size_type size_type;

                   SymbolTable         ();

                   SymbolTable         (SymbolTable const&)=delete;

    SymbolTable&   operator=           (SymbolTable const&)=delete;

                   ~SymbolTable        ();

    void           pushScope           ();

    void           popScope            ();

    size_type      scopeLevel          () const;

    size_type      scopeLevel          (String const& name) const;

    void           addDefinition       (NameVertex* definition);

    bool           hasDefinition       (String const& name) const;

    NameVertex const* definition       (String const& name) const;

    NameVertex*    definition          (String const& name);

    bool           empty               () const;

    size_type      size                () const;

private:

    //! Definitions by name.
    std::map<String, Definitions> _definitions;

    //! Definitions by scope level.
    std::vector<Definitions> _scopes;

    Definitions const& definitions     (String const& name) const;

    Definitions&   definitions         (String const& name);

};

} // namespace language
} // namespace ranally
