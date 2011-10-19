#ifndef INCLUDED_RANALLY_LANGUAGE_SYMBOLTABLE
#define INCLUDED_RANALLY_LANGUAGE_SYMBOLTABLE

#include <list>
#include <map>
#include <vector>
#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>



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
class SymbolTable: private boost::noncopyable
{

  friend class SymbolTableTest;

public:

  //! Type for lists of definitions.
  typedef std::list<NameVertex*> Definitions;

  //! Type for scope levels.
  typedef std::vector<Definitions>::size_type size_type;

                   SymbolTable         ();

                   ~SymbolTable        ();

  void             pushScope           ();

  void             popScope            ();

  size_type        scopeLevel          () const;

  size_type        scopeLevel          (UnicodeString const& name) const;

  void             addDefinition       (NameVertex* definition);

  bool             hasDefinition       (UnicodeString const& name) const;

  NameVertex const* definition         (UnicodeString const& name) const;

  NameVertex*      definition          (UnicodeString const& name);

  bool             empty               () const;

  size_type        size                () const;

protected:

private:

  //! Definitions by name.
  std::map<UnicodeString, Definitions> _definitions;

  //! Definitions by scope level.
  std::vector<Definitions> _scopes;

  Definitions const& definitions       (UnicodeString const& name) const;

  Definitions&     definitions         (UnicodeString const& name);

};

} // namespace language
} // namespace ranally

#endif
