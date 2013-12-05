#pragma once
#include "fern/ast/core/expression_vertex.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class StringVertex:
    public ExpressionVertex
{

    friend class StringVertexTest;

public:

    LOKI_DEFINE_VISITABLE()

                   StringVertex        (int line_nr,
                                        int col_id,
                                        String const& value);

                   ~StringVertex       ()=default;

                   StringVertex        (StringVertex&&)=delete;

    StringVertex&  operator=           (StringVertex&&)=delete;

                   StringVertex        (StringVertex const&)=delete;

    StringVertex&  operator=           (StringVertex const&)=delete;

    String const&  value               () const;

private:

    String         _value;

};

} // namespace fern