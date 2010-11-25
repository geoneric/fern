#ifndef INCLUDED_RANALLY_NAMEVERTEX
#define INCLUDED_RANALLY_NAMEVERTEX

#include "ExpressionVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class NameVertex: public ExpressionVertex
{

  friend class NameVertexTest;

private:

  //! Definition vertex (left side of an assignment).
  NameVertex*      _definition;

protected:

public:

  LOKI_DEFINE_VISITABLE()

                   NameVertex          (UnicodeString const& name);

                   NameVertex          (int lineNr,
                                        int colId,
                                        UnicodeString const& name);

  /* virtual */    ~NameVertex         ();

  void             setDefinition       (NameVertex* definition);

  NameVertex const* definition         () const;

};

} // namespace ranally

#endif
