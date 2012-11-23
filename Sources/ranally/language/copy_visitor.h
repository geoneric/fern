#pragma once
#include "ranally/language/visitor.h"


namespace ranally {

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

                   CopyVisitor         ()=default;

                   ~CopyVisitor        ()=default;

                   CopyVisitor         (CopyVisitor&&)=delete;

    CopyVisitor&   operator=           (CopyVisitor&&)=delete;

                   CopyVisitor         (CopyVisitor const&)=delete;

    CopyVisitor&   operator=           (CopyVisitor const&)=delete;

    std::shared_ptr<ScriptVertex> const& scriptVertex() const;

    // SyntaxVertices const& syntaxVertices () const;

    // StatementVertices const& statements  () const;

private:

    std::shared_ptr<ScriptVertex> _scriptVertex;

    // SyntaxVertices   _syntaxVertices;

    StatementVertices _statements;

    std::shared_ptr<StatementVertex> _statementVertex;

    void           visitStatements     (StatementVertices& statements);

    void           Visit               (AssignmentVertex&);

    void           Visit               (FunctionVertex&);

    void           Visit               (IfVertex&);

    void           Visit               (NameVertex&);

    void           Visit               (OperatorVertex&);

    void           Visit               (ScriptVertex&);

    void           Visit               (WhileVertex&);

};

} // namespace ranally
