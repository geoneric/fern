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


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<typename T>
class NumberVertex:
    public ExpressionVertex
{

    friend class NumberVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   NumberVertex        (T value);

                   NumberVertex        (int line_nr,
                                        int col_id,
                                        T value);

                   ~NumberVertex       ()=default;

                   NumberVertex        (NumberVertex&&)=delete;

    NumberVertex&  operator=           (NumberVertex&&)=delete;

                   NumberVertex        (NumberVertex const&)=delete;

    NumberVertex&  operator=           (NumberVertex const&)=delete;

    T              value               () const;

private:

    T              _value;

};

} // namespace fern
