// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/ast/core/sentinel_vertex.h"


namespace fern {

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

} // namespace fern
