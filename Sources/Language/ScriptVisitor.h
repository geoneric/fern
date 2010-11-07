#ifndef INCLUDED_RANALLY_SCRIPTVISITOR
#define INCLUDED_RANALLY_SCRIPTVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>
#include <unicode/unistr.h>

#include "SyntaxVertex.h"



namespace ranally {

class AssignmentVertex;
class FunctionVertex;
class IfVertex;
class NameVertex;
template<typename T>
  class NumberVertex;
class OperatorVertex;
class ScriptVertex;
class StringVertex;
class WhileVertex;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex, UnicodeString>,
  public Loki::Visitor<FunctionVertex, UnicodeString>,
  public Loki::Visitor<IfVertex, UnicodeString>,
  public Loki::Visitor<NameVertex, UnicodeString>,
  public Loki::Visitor<NumberVertex<long>, UnicodeString>,
  public Loki::Visitor<NumberVertex<long long>, UnicodeString>,
  public Loki::Visitor<NumberVertex<double>, UnicodeString>,
  public Loki::Visitor<OperatorVertex, UnicodeString>,
  public Loki::Visitor<ScriptVertex, UnicodeString>,
  public Loki::Visitor<StringVertex, UnicodeString>,
  public Loki::Visitor<SyntaxVertex, UnicodeString>,
  public Loki::Visitor<WhileVertex, UnicodeString>
{

  friend class ScriptVisitorTest;

private:

  size_t           _tabSize;

  size_t           _indentLevel;

  UnicodeString    indent              (UnicodeString const& statement);

  UnicodeString    visitStatements     (StatementVertices const& statements);

  UnicodeString    visitExpressions    (ExpressionVertices const& expressions);

protected:

public:

                   ScriptVisitor       (size_t tabSize=2);

  /* virtual */    ~ScriptVisitor      ();

  UnicodeString    Visit               (AssignmentVertex&);

  UnicodeString    Visit               (FunctionVertex&);

  UnicodeString    Visit               (IfVertex&);

  UnicodeString    Visit               (NameVertex&);

  UnicodeString    Visit               (NumberVertex<long>&);

  UnicodeString    Visit               (NumberVertex<long long>&);

  UnicodeString    Visit               (NumberVertex<double>&);

  UnicodeString    Visit               (OperatorVertex&);

  UnicodeString    Visit               (ScriptVertex&);

  UnicodeString    Visit               (StringVertex&);

  UnicodeString    Visit               (SyntaxVertex&);

  UnicodeString    Visit               (WhileVertex&);

};

} // namespace ranally

#endif
