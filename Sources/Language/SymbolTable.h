#ifndef INCLUDED_RANALLY_SYMBOLTABLE
#define INCLUDED_RANALLY_SYMBOLTABLE

#include <list>
#include <map>
#include <stack>
#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>



namespace ranally {

class Definition;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class SymbolTable: private boost::noncopyable
{

  friend class SymbolTableTest;

public:

  typedef std::list<Definition*> Definitions;

private:

  //! Definitions by identifier name.
  std::map<UnicodeString, Definitions> _definitions;

  //! Definitions by scope level.
  std::stack<Definitions> _scopes;

  Definitions const& definitions       (UnicodeString const& name) const;

  Definitions&     definitions         (UnicodeString const& name);

protected:

public:

  typedef std::stack<Definitions>::size_type size_type;

                   SymbolTable         ();

  /* virtual */    ~SymbolTable        ();

  void             pushScope           ();

  void             popScope            ();

  void             addDefinition       (Definition* definition);

  Definition const& definition         (UnicodeString const& name) const;

  Definition&      definition          (UnicodeString const& name);

};

} // namespace ranally

#endif
