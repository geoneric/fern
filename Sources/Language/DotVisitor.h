#ifndef INCLUDED_RANALLY_DOTVISITOR
#define INCLUDED_RANALLY_DOTVISITOR

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
class DotVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex, UnicodeString>,
  public Loki::Visitor<FunctionVertex, UnicodeString>,
  public Loki::Visitor<IfVertex, UnicodeString>,
  public Loki::Visitor<NameVertex, UnicodeString>,
  public Loki::Visitor<NumberVertex<int8_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<int16_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<int32_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<int64_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<uint8_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<uint16_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<uint32_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<uint64_t>, UnicodeString>,
  public Loki::Visitor<NumberVertex<float>, UnicodeString>,
  public Loki::Visitor<NumberVertex<double>, UnicodeString>,
  public Loki::Visitor<OperatorVertex, UnicodeString>,
  public Loki::Visitor<ScriptVertex, UnicodeString>,
  public Loki::Visitor<StringVertex, UnicodeString>,
  public Loki::Visitor<SyntaxVertex, UnicodeString>,
  public Loki::Visitor<WhileVertex, UnicodeString>
{

  friend class DotVisitorTest;

private:

  size_t           _tabSize;

  size_t           _indentLevel;

  UnicodeString    indent              (UnicodeString const& statement);

  UnicodeString    visitStatements     (StatementVertices const& statements);

  UnicodeString    visitExpressions    (ExpressionVertices const& expressions);

protected:

public:

                   DotVisitor          ();

  /* virtual */    ~DotVisitor         ();

  UnicodeString    Visit               (AssignmentVertex&);

  UnicodeString    Visit               (FunctionVertex&);

  UnicodeString    Visit               (IfVertex&);

  UnicodeString    Visit               (NameVertex&);

  UnicodeString    Visit               (NumberVertex<int8_t>&);

  UnicodeString    Visit               (NumberVertex<int16_t>&);

  UnicodeString    Visit               (NumberVertex<int32_t>&);

  UnicodeString    Visit               (NumberVertex<int64_t>&);

  UnicodeString    Visit               (NumberVertex<uint8_t>&);

  UnicodeString    Visit               (NumberVertex<uint16_t>&);

  UnicodeString    Visit               (NumberVertex<uint32_t>&);

  UnicodeString    Visit               (NumberVertex<uint64_t>&);

  UnicodeString    Visit               (NumberVertex<float>&);

  UnicodeString    Visit               (NumberVertex<double>&);

  UnicodeString    Visit               (OperatorVertex&);

  UnicodeString    Visit               (ScriptVertex&);

  UnicodeString    Visit               (StringVertex&);

  UnicodeString    Visit               (SyntaxVertex&);

  UnicodeString    Visit               (WhileVertex&);

};

} // namespace ranally

#endif
