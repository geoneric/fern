#pragma once
#include "ranally/ast/core/sentinel_vertex.h"


namespace ranally {

class ScopeVertex:
    public AstVertex
{

public:

                   ScopeVertex         (StatementVertices const& statements);

                   ~ScopeVertex        ()=default;

                   ScopeVertex         (ScopeVertex&&)=delete;

    ScopeVertex&   operator=           (ScopeVertex&&)=delete;

                   ScopeVertex         (ScopeVertex const&)=delete;

    ScopeVertex&   operator=           (ScopeVertex const&)=delete;

    StatementVertices const& statements() const;

    StatementVertices& statements      ();

    std::shared_ptr<SentinelVertex> const& sentinel() const;

    std::shared_ptr<SentinelVertex>& sentinel();

private:

    StatementVertices _statements;

    std::shared_ptr<SentinelVertex> _sentinel;

};

} // namespace ranally
