#ifndef INCLUDED_RANALLY_LANGUAGE_VISITOR
#define INCLUDED_RANALLY_LANGUAGE_VISITOR

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
*/
class Visitor:
  private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<AssignmentVertex>,
  public Loki::Visitor<FunctionVertex>,
  public Loki::Visitor<IfVertex>,
  public Loki::Visitor<NameVertex>,
  public Loki::Visitor<language::NumberVertex<int8_t> >,
  public Loki::Visitor<language::NumberVertex<int16_t> >,
  public Loki::Visitor<language::NumberVertex<int32_t> >,
  public Loki::Visitor<language::NumberVertex<int64_t> >,
  public Loki::Visitor<language::NumberVertex<uint8_t> >,
  public Loki::Visitor<language::NumberVertex<uint16_t> >,
  public Loki::Visitor<language::NumberVertex<uint32_t> >,
  public Loki::Visitor<language::NumberVertex<uint64_t> >,
  public Loki::Visitor<language::NumberVertex<float> >,
  public Loki::Visitor<language::NumberVertex<double> >,
  public Loki::Visitor<OperatorVertex>,
  public Loki::Visitor<ScriptVertex>,
  public Loki::Visitor<StringVertex>,
  public Loki::Visitor<WhileVertex>
{

  friend class VisitorTest;

public:

protected:

                   Visitor             ();

  virtual          ~Visitor            ();

  virtual void     visitStatements     (StatementVertices const& statements);

  virtual void     visitExpressions    (ExpressionVertices const& expressions);

private:

  virtual void     Visit               (AssignmentVertex& vertex);

  virtual void     Visit               (FunctionVertex& vertex);

  virtual void     Visit               (IfVertex& vertex);

  virtual void     Visit               (NameVertex& vertex);

  virtual void     Visit               (language::NumberVertex<int8_t>& vertex);

  virtual void     Visit               (
                                  language::NumberVertex<int16_t>& vertex);

  virtual void     Visit               (
                                  language::NumberVertex<int32_t>& vertex);

  virtual void     Visit               (
                                  language::NumberVertex<int64_t>& vertex);

  virtual void     Visit               (
                                  language::NumberVertex<uint8_t>& vertex);

  virtual void     Visit               (
                                  language::NumberVertex<uint16_t>& vertex);

  virtual void     Visit               (
                                  language::NumberVertex<uint32_t>& vertex);

  virtual void     Visit               (
                                  language::NumberVertex<uint64_t>& vertex);

  virtual void     Visit               (language::NumberVertex<float>& vertex);

  virtual void     Visit               (language::NumberVertex<double>& vertex);


  virtual void     Visit               (OperatorVertex& vertex);

  virtual void     Visit               (ScriptVertex& vertex);

  virtual void     Visit               (StringVertex& vertex);

  virtual void     Visit               (WhileVertex& vertex);

};

} // namespace language
} // namespace ranally

#endif
