#pragma once
#include "ranally/language/expression_vertex.h"


namespace ranally {

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

private:

    T              _value;

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

};

} // namespace ranally
