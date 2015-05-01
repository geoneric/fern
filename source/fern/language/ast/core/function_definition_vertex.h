// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/core/scope_vertex.h"
#include "fern/language/ast/core/statement_vertex.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FunctionDefinitionVertex:
    public StatementVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   FunctionDefinitionVertex(
                                  std::string const& name,
                                  ExpressionVertices const& arguments,
                                  std::shared_ptr<ScopeVertex> const& scope);

                   ~FunctionDefinitionVertex()=default;

                   FunctionDefinitionVertex(
                                        FunctionDefinitionVertex&&)=delete;

    FunctionDefinitionVertex& operator=(FunctionDefinitionVertex&&)=delete;

                   FunctionDefinitionVertex(
                                        FunctionDefinitionVertex const&)=delete;

    FunctionDefinitionVertex& operator=(FunctionDefinitionVertex const&)=delete;

    std::string const&
                   name                () const;

    ExpressionVertices const& arguments() const;

    ExpressionVertices& arguments      ();

    std::shared_ptr<ScopeVertex> const& scope() const;

    std::shared_ptr<ScopeVertex>& scope();

private:

    std::string    _name;

    ExpressionVertices _arguments;

    std::shared_ptr<ScopeVertex> _scope;

};

} // namespace language
} // namespace fern
