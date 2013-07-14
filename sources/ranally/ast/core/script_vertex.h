#pragma once
#include "ranally/ast/core/scope_vertex.h"
#include "ranally/ast/core/statement_vertex.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScriptVertex:
    public SyntaxVertex
{

    friend class ScriptVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   ScriptVertex        (
                                  String const& source_name,
                                  std::shared_ptr<ScopeVertex> const& scope);

                   ~ScriptVertex       ()=default;

                   ScriptVertex        (ScriptVertex&&)=delete;

    ScriptVertex&  operator=           (ScriptVertex&&)=delete;

                   ScriptVertex        (ScriptVertex const&)=delete;

    ScriptVertex&  operator=           (ScriptVertex const&)=delete;

    String const&  source_name         () const;

    std::shared_ptr<ScopeVertex> const& scope() const;

    std::shared_ptr<ScopeVertex>& scope();

private:

    String         _source_name;

    std::shared_ptr<ScopeVertex> _scope;

};

typedef std::shared_ptr<ScriptVertex> ScriptVertexPtr;

// bool               operator==          (ScriptVertex const& lhs,
//                                         ScriptVertex const& rhs);
// 
// bool               operator!=          (ScriptVertex const& lhs,
//                                         ScriptVertex const& rhs);

} // namespace ranally
