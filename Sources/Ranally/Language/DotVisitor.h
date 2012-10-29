#pragma once
#include "Ranally/Language/Visitor.h"
#include "Ranally/Util/String.h"


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

  String const&    script              () const;

protected:

                   DotVisitor          ();

  void             setScript           (String const& string);

  void             addScript           (String const& string);

private:

  String           _script;

  virtual void     Visit               (language::ScriptVertex& vertex)=0;

};

} // namespace ranally
