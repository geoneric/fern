#ifndef INCLUDED_RANALLY_LANGUAGE_SCRIPTVISITOR
#define INCLUDED_RANALLY_LANGUAGE_SCRIPTVISITOR

#include <unicode/unistr.h>
#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVisitor
  : public Visitor
{

  friend class ScriptVisitorTest;

public:

                   ScriptVisitor       (size_t tabSize=2);

                   ~ScriptVisitor      ();

  UnicodeString const& script          () const;

private:

  size_t           _tabSize;

  size_t           _indentLevel;

  UnicodeString    _script;

  // void             indent              (UnicodeString const& statement);

  UnicodeString    indentation         () const;

  void             visitStatements     (
                                  language::StatementVertices& statements);

  void             visitExpressions    (
                             language::ExpressionVertices const& expressions);

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

};

} // namespace language
} // namespace ranally

#endif
