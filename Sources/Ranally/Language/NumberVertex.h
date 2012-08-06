#ifndef INCLUDED_RANALLY_LANGUAGE_NUMBERVERTEX
#define INCLUDED_RANALLY_LANGUAGE_NUMBERVERTEX

#include "Ranally/Language/ExpressionVertex.h"



namespace ranally {
namespace language {

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

  T                _value;

public:

  LOKI_DEFINE_VISITABLE()

                   NumberVertex        (T value);

                   NumberVertex        (int lineNr,
                                        int colId,
                                        T value);

                   ~NumberVertex       ();

  T                value               () const;

};

} // namespace language
} // namespace ranally

#endif
