#ifndef INCLUDED_RANALLY_LANGUAGE_SCRIPTVISITOR
#define INCLUDED_RANALLY_LANGUAGE_SCRIPTVISITOR

#include <boost/noncopyable.hpp>
#include <loki/Visitor.h>
#include <unicode/unistr.h>
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

} // namespace language

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVisitor: private boost::noncopyable,
  public Loki::BaseVisitor,
  public Loki::Visitor<language::AssignmentVertex>,
  public Loki::Visitor<language::FunctionVertex>,
  public Loki::Visitor<language::IfVertex>,
  public Loki::Visitor<language::NameVertex>,
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
  public Loki::Visitor<language::OperatorVertex>,
  public Loki::Visitor<language::ScriptVertex>,
  public Loki::Visitor<language::StringVertex>,
  public Loki::Visitor<language::SyntaxVertex>,
  public Loki::Visitor<language::WhileVertex>
{

  friend class ScriptVisitorTest;

public:

                   ScriptVisitor       (size_t tabSize=2);

                   ~ScriptVisitor      ();

  UnicodeString const& script          () const;

  void             Visit               (language::AssignmentVertex&);

  void             Visit               (language::FunctionVertex&);

  void             Visit               (language::IfVertex&);

  void             Visit               (language::NameVertex&);

  void             Visit               (language::NumberVertex<int8_t>&);

  void             Visit               (language::NumberVertex<int16_t>&);

  void             Visit               (language::NumberVertex<int32_t>&);

  void             Visit               (language::NumberVertex<int64_t>&);

  void             Visit               (language::NumberVertex<uint8_t>&);

  void             Visit               (language::NumberVertex<uint16_t>&);

  void             Visit               (language::NumberVertex<uint32_t>&);

  void             Visit               (language::NumberVertex<uint64_t>&);

  void             Visit               (language::NumberVertex<float>&);

  void             Visit               (language::NumberVertex<double>&);

  void             Visit               (language::OperatorVertex&);

  void             Visit               (language::ScriptVertex&);

  void             Visit               (language::StringVertex&);

  void             Visit               (language::SyntaxVertex&);

  void             Visit               (language::WhileVertex&);

private:

  size_t           _tabSize;

  size_t           _indentLevel;

  UnicodeString    _script;

  // void             indent              (UnicodeString const& statement);

  UnicodeString    indentation         () const;

  void             visitStatements     (
                             language::StatementVertices const& statements);

  void             visitExpressions    (
                             language::ExpressionVertices const& expressions);

};

} // namespace ranally

#endif
