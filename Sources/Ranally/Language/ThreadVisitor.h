#ifndef INCLUDED_RANALLY_LANGUAGE_THREADVISITOR
#define INCLUDED_RANALLY_LANGUAGE_THREADVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>
#include "Ranally/Language/SyntaxVertex.h"



namespace ranally {
namespace language {

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
  \todo      Is it possible to pass pointers as arguments to the Visit functions?
*/
class ThreadVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex>,
  public Loki::Visitor<FunctionVertex>,
  public Loki::Visitor<IfVertex>,
  public Loki::Visitor<NameVertex>,
  public Loki::Visitor<NumberVertex<int8_t>>,
  public Loki::Visitor<NumberVertex<int16_t>>,
  public Loki::Visitor<NumberVertex<int32_t>>,
  public Loki::Visitor<NumberVertex<int64_t>>,
  public Loki::Visitor<NumberVertex<uint8_t>>,
  public Loki::Visitor<NumberVertex<uint16_t>>,
  public Loki::Visitor<NumberVertex<uint32_t>>,
  public Loki::Visitor<NumberVertex<uint64_t>>,
  public Loki::Visitor<NumberVertex<float>>,
  public Loki::Visitor<NumberVertex<double>>,
  public Loki::Visitor<OperatorVertex>,
  public Loki::Visitor<ScriptVertex>,
  public Loki::Visitor<StringVertex>,
  public Loki::Visitor<SyntaxVertex>,
  public Loki::Visitor<WhileVertex>
{

  friend class ThreadVisitorTest;

private:

  //! Last vertex processed on the control flow path.
  SyntaxVertex*    _lastVertex;

  void             visitStatements     (StatementVertices const& statements);

  void             visitExpressions    (ExpressionVertices const& expressions);

  template<typename T>
  void             Visit               (NumberVertex<T>&);

protected:

public:

                   ThreadVisitor       ();

  /* virtual */    ~ThreadVisitor      ();

  void             Visit               (AssignmentVertex&);

  void             Visit               (FunctionVertex&);

  void             Visit               (IfVertex&);

  void             Visit               (NameVertex&);

  void             Visit               (NumberVertex<int8_t>&);

  void             Visit               (NumberVertex<int16_t>&);

  void             Visit               (NumberVertex<int32_t>&);

  void             Visit               (NumberVertex<int64_t>&);

  void             Visit               (NumberVertex<uint8_t>&);

  void             Visit               (NumberVertex<uint16_t>&);

  void             Visit               (NumberVertex<uint32_t>&);

  void             Visit               (NumberVertex<uint64_t>&);

  void             Visit               (NumberVertex<float>&);

  void             Visit               (NumberVertex<double>&);

  void             Visit               (OperatorVertex&);

  void             Visit               (ScriptVertex&);

  void             Visit               (StringVertex&);

  void             Visit               (SyntaxVertex&);

  void             Visit               (WhileVertex&);

};

} // namespace language
} // namespace ranally

#endif
