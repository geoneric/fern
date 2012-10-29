#pragma once
#include "Ranally/Language/Visitor.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class CopyVisitor:
  public Visitor
{

  friend class CopyVisitorTest;

public:

                   CopyVisitor         ();

                   ~CopyVisitor        ();

  boost::shared_ptr<ScriptVertex> const& scriptVertex() const;

  // SyntaxVertices const& syntaxVertices () const;

  // StatementVertices const& statements  () const;

private:

  boost::shared_ptr<ScriptVertex> _scriptVertex;

  // SyntaxVertices   _syntaxVertices;

  StatementVertices _statements;

  boost::shared_ptr<StatementVertex> _statementVertex;

  void             visitStatements     (StatementVertices& statements);

  void             Visit               (AssignmentVertex&);

  void             Visit               (FunctionVertex&);

  void             Visit               (IfVertex&);

  void             Visit               (NameVertex&);

  void             Visit               (OperatorVertex&);

  void             Visit               (ScriptVertex&);

  void             Visit               (WhileVertex&);

};

} // namespace language
} // namespace ranally
