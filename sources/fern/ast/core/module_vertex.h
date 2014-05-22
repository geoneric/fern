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
