#pragma once
#include "ranally/core/string.h"
#include "ranally/ast/visitor/visitor.h"


namespace ranally {

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

                   ScriptVisitor       (size_t tab_size=4);

                   ~ScriptVisitor      ()=default;

                   ScriptVisitor       (ScriptVisitor&&)=delete;

    ScriptVisitor& operator=           (ScriptVisitor&&)=delete;

                   ScriptVisitor       (ScriptVisitor const&)=delete;

    ScriptVisitor& operator=           (ScriptVisitor const&)=delete;

    String const&  script              () const;

private:

    size_t         _tab_size;

    size_t         _indent_level;

    String         _script;

    // void           indent              (String const& statement);

    String         indentation         () const;

    void           visit_statements    (StatementVertices& statements);

    void           visit_expressions   (ExpressionVertices const& expressions);

    void           Visit               (AssignmentVertex&);

    void           Visit               (FunctionVertex&);

    void           Visit               (IfVertex&);

    void           Visit               (NameVertex&);

    void           Visit               (NumberVertex<int8_t>&);

    void           Visit               (NumberVertex<int16_t>&);

    void           Visit               (NumberVertex<int32_t>&);

    void           Visit               (NumberVertex<int64_t>&);

    void           Visit               (NumberVertex<uint8_t>&);

    void           Visit               (NumberVertex<uint16_t>&);

    void           Visit               (NumberVertex<uint32_t>&);

    void           Visit               (NumberVertex<uint64_t>&);

    void           Visit               (NumberVertex<float>&);

    void           Visit               (NumberVertex<double>&);

    void           Visit               (OperatorVertex&);

    void           Visit               (ScriptVertex&);

    void           Visit               (StringVertex&);

    void           Visit               (SubscriptVertex&);

    void           Visit               (SyntaxVertex&);

    void           Visit               (WhileVertex&);

};

} // namespace ranally
