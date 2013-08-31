#pragma once
#include "geoneric/ast/core/ast_vertex.h"


namespace geoneric {

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

} // namespace geoneric
