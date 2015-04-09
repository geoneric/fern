// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/ast/core/scope_vertex.h"
#include "fern/ast/core/statement_vertex.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ModuleVertex:
    public AstVertex
{

public:

    LOKI_DEFINE_VISITABLE()

                   ModuleVertex        (
                                  String const& source_name,
                                  std::shared_ptr<ScopeVertex> const& scope);

                   ~ModuleVertex       ()=default;

                   ModuleVertex        (ModuleVertex&&)=delete;

    ModuleVertex&  operator=           (ModuleVertex&&)=delete;

                   ModuleVertex        (ModuleVertex const&)=delete;

    ModuleVertex&  operator=           (ModuleVertex const&)=delete;

    String const&  source_name         () const;

    std::shared_ptr<ScopeVertex> const& scope() const;

    std::shared_ptr<ScopeVertex>& scope();

private:

    String         _source_name;

    std::shared_ptr<ScopeVertex> _scope;

};

using ModuleVertexPtr = std::shared_ptr<ModuleVertex>;

// bool               operator==          (ModuleVertex const& lhs,
//                                         ModuleVertex const& rhs);
// 
// bool               operator!=          (ModuleVertex const& lhs,
//                                         ModuleVertex const& rhs);

} // namespace fern
