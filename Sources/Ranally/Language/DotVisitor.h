#ifndef INCLUDED_RANALLY_LANGUAGE_DOTVISITOR
#define INCLUDED_RANALLY_LANGUAGE_DOTVISITOR

#include "Ranally/Language/Visitor.h"
#include <unicode/unistr.h>



namespace ranally {
namespace language {

template<typename T>
  class NumberVertex;

} // namespace language



//! Base class for visitors emitting dot graphs.
/*!
  The dot graphs are useful for debugging purposes. The graphs are handy
  for visualising the syntax-tree.
*/
class DotVisitor:
  public language::Visitor
{

  friend class DotVisitorTest;

public:

  virtual          ~DotVisitor         ();

  UnicodeString const& script          () const;

protected:

                   DotVisitor          ();

  void             setScript           (UnicodeString const& string);

  void             addScript           (UnicodeString const& string);

private:

  UnicodeString    _script;

  virtual void     Visit               (language::ScriptVertex& vertex)=0;

};

} // namespace ranally

#endif
