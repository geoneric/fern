#pragma once
#include <map>
#include "ranally/language/name_vertex.h"
#include "ranally/language/visitor.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \todo      Only constant expressions can be inlined!!! Currently, all
             expressions are inlined, which is wrong.
  \todo      Rename this class to InlineConstantExpressionVisitor, or similar.
  \todo      Not sure if a visitor is the way to go. This visitor updates the
             shape of the syntax tree, which is currently visited! Maybe a new
             tree should be created?
  \sa        .
*/
class OptimizeVisitor
    : public Visitor
{

    friend class OptimizeVisitorTest;

public:

                   OptimizeVisitor     ();

                   ~OptimizeVisitor    ()=default;

                   OptimizeVisitor     (OptimizeVisitor&&)=delete;

    OptimizeVisitor& operator=         (OptimizeVisitor&&)=delete;

                   OptimizeVisitor     (OptimizeVisitor const&)=delete;

    OptimizeVisitor& operator=         (OptimizeVisitor const&)=delete;

private:

    enum Mode {
        Defining,
        Using
    };

    Mode           _mode;

    std::map<ExpressionVertex const*, ExpressionVertexPtr> _inline_expressions;

    std::vector<ExpressionVertexPtr> _inlined_expressions;

    std::vector<StatementVertex*> _superfluous_statements;

    void           register_expression_for_inlining(
                                        ExpressionVertex const* use,
                                        ExpressionVertexPtr const& expression);

    void           visit_statements    (StatementVertices& statements);

    void           Visit               (AssignmentVertex& vertex);

    void           Visit               (NameVertex& vertex);

    void           Visit               (ScriptVertex& vertex);

};

} // namespace ranally
