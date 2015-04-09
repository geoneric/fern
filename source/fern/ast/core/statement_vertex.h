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

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class StatementVertex:
    public AstVertex
{

    friend class StatementVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

    virtual        ~StatementVertex    ()=default;

                   StatementVertex     (StatementVertex&&)=delete;

    StatementVertex& operator=         (StatementVertex&&)=delete;

                   StatementVertex     (StatementVertex const&)=delete;

    StatementVertex& operator=         (StatementVertex const&)=delete;

protected:

                   StatementVertex     ()=default;

                   StatementVertex     (int lineNr,
                                        int colId);

private:

};

} // namespace fern
