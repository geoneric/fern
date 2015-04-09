// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/ast/core/ast_vertex.h"


namespace fern {

class SentinelVertex:
    public AstVertex
{

public:

                   SentinelVertex      ();

                   ~SentinelVertex     ()=default;

                   SentinelVertex      (SentinelVertex&&)=delete;

    SentinelVertex& operator=          (SentinelVertex&&)=delete;

                   SentinelVertex      (SentinelVertex const&)=delete;

    SentinelVertex& operator=          (SentinelVertex const&)=delete;

};

} // namespace fern
