#ifndef INCLUDED_RANALLY_SCRIPTVISITOR
#define INCLUDED_RANALLY_SCRIPTVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>



namespace ranally {

class SyntaxVertex;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVisitor: private boost::noncopyable,
                     public Loki::BaseVisitor,
                     public Loki::Visitor<SyntaxVertex>
{

  friend class ScriptVisitorTest;

private:

protected:

public:

                   ScriptVisitor       ();

  /* virtual */    ~ScriptVisitor      ();

  void             Visit               (SyntaxVertex&);

};

} // namespace ranally

#endif
