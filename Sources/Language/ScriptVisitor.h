#ifndef INCLUDED_RANALLY_SCRIPTVISITOR
#define INCLUDED_RANALLY_SCRIPTVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>
#include <unicode/unistr.h>



namespace ranally {

class AssignmentVertex;
class FunctionVertex;
class NameVertex;
class ScriptVertex;
class StringVertex;
class SyntaxVertex;
template<typename T>
  class NumberVertex;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex, UnicodeString>,
  public Loki::Visitor<FunctionVertex, UnicodeString>,
  public Loki::Visitor<NameVertex, UnicodeString>,
  public Loki::Visitor<NumberVertex<int>, UnicodeString>,
  public Loki::Visitor<NumberVertex<long long>, UnicodeString>,
  public Loki::Visitor<NumberVertex<double>, UnicodeString>,
  public Loki::Visitor<ScriptVertex, UnicodeString>,
  public Loki::Visitor<StringVertex, UnicodeString>,
  public Loki::Visitor<SyntaxVertex, UnicodeString>
{

  friend class ScriptVisitorTest;

private:

protected:

public:

                   ScriptVisitor       ();

  /* virtual */    ~ScriptVisitor      ();

  UnicodeString    Visit               (AssignmentVertex&);

  UnicodeString    Visit               (FunctionVertex&);

  UnicodeString    Visit               (NameVertex&);

  UnicodeString    Visit               (NumberVertex<int>&);

  UnicodeString    Visit               (NumberVertex<long long>&);

  UnicodeString    Visit               (NumberVertex<double>&);

  UnicodeString    Visit               (ScriptVertex&);

  UnicodeString    Visit               (StringVertex&);

  UnicodeString    Visit               (SyntaxVertex&);

};

} // namespace ranally

#endif
