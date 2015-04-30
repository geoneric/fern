// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/core/expression_vertex.h"
#include "fern/language/ast/core/scope_vertex.h"
#include "fern/language/ast/core/statement_vertex.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class WhileVertex:
    public StatementVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   WhileVertex         (
                        std::shared_ptr<ExpressionVertex> const& condition,
                        std::shared_ptr<ScopeVertex> const& true_scope,
                        std::shared_ptr<ScopeVertex> const& false_scope);

                   ~WhileVertex        ()=default;

                   WhileVertex         (WhileVertex&&)=delete;

    WhileVertex&   operator=           (WhileVertex&&)=delete;

                   WhileVertex         (WhileVertex const&)=delete;

    WhileVertex&   operator=           (WhileVertex const&)=delete;

    std::shared_ptr<ExpressionVertex> const& condition() const;

    std::shared_ptr<ScopeVertex> const& true_scope() const;

    std::shared_ptr<ScopeVertex>& true_scope ();

    std::shared_ptr<ScopeVertex> const& false_scope() const;

    std::shared_ptr<ScopeVertex>& false_scope();

    std::shared_ptr<SentinelVertex> const& sentinel() const;

    std::shared_ptr<SentinelVertex>& sentinel();

private:

    ExpressionVertexPtr _condition;

    std::shared_ptr<ScopeVertex> _true_scope;

    std::shared_ptr<ScopeVertex> _false_scope;

    std::shared_ptr<SentinelVertex> _sentinel;

};

} // namespace language
} // namespace fern
