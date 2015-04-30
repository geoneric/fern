// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/string.h"
#include "fern/language/ast/visitor/ast_visitor.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ModuleVisitor:
    public AstVisitor
{

public:

                   ModuleVisitor       (size_t tab_size=4);

                   ~ModuleVisitor      ()=default;

                   ModuleVisitor       (ModuleVisitor&&)=delete;

    ModuleVisitor& operator=           (ModuleVisitor&&)=delete;

                   ModuleVisitor       (ModuleVisitor const&)=delete;

    ModuleVisitor& operator=           (ModuleVisitor const&)=delete;

    String const&  module              () const;

private:

    size_t         _tab_size;

    size_t         _indent_level;

    String         _module;

    // void           indent              (String const& statement);

    String         indentation         () const;

    void           visit_statements    (StatementVertices& statements);

    void           visit_expressions   (ExpressionVertices const& expressions);

    void           Visit               (AssignmentVertex&);

    void           Visit               (AttributeVertex& vertex);

    void           Visit               (FunctionCallVertex&);

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

    void           Visit               (ModuleVertex&);

    void           Visit               (StringVertex&);

    void           Visit               (SubscriptVertex&);

    void           Visit               (AstVertex&);

    void           Visit               (WhileVertex&);

};

} // namespace fern
