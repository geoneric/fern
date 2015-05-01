// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/visitor/ast_visitor.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class CopyVisitor:
    public AstVisitor
{

    friend class CopyVisitorTest;

public:

                   CopyVisitor         ()=default;

                   ~CopyVisitor        ()=default;

                   CopyVisitor         (CopyVisitor&&)=delete;

    CopyVisitor&   operator=           (CopyVisitor&&)=delete;

                   CopyVisitor         (CopyVisitor const&)=delete;

    CopyVisitor&   operator=           (CopyVisitor const&)=delete;

    std::shared_ptr<ModuleVertex> const& module_vertex() const;

    // SyntaxVertices const& syntax_vertices () const;

    // StatementVertices const& statements  () const;

private:

    std::shared_ptr<ModuleVertex> _module_vertex;

    // SyntaxVertices   _syntax_vertices;

    StatementVertices _statements;

    std::shared_ptr<StatementVertex> _statement_vertex;

    void           visit_statements    (StatementVertices& statements);

    void           Visit               (AssignmentVertex&);

    void           Visit               (FunctionCallVertex&);

    void           Visit               (IfVertex&);

    void           Visit               (NameVertex&);

    void           Visit               (OperatorVertex&);

    void           Visit               (ModuleVertex&);

    void           Visit               (WhileVertex&);

};

} // namespace language
} // namespace fern
