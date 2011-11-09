#ifndef INCLUDED_RANALLY_LANGUAGE_NAMEVERTEX
#define INCLUDED_RANALLY_LANGUAGE_NAMEVERTEX

#include "Ranally/Language/ExpressionVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class NameVertex: public ExpressionVertex
{

  friend class NameVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

                   NameVertex          (UnicodeString const& name);

                   NameVertex          (int lineNr,
                                        int colId,
                                        UnicodeString const& name);

                   ~NameVertex         ();

  // void             setDefinition       (NameVertex* definition);

  // NameVertex const* definition         () const;

  void             addDefinition       (NameVertex* vertex);

  std::vector<NameVertex*> const& definitions() const;

  // NameVertex*      definition          ();

  void             addUse              (NameVertex* vertex);

  std::vector<NameVertex*> const& uses () const;

private:

  // //! Definition of the name (left side of an assignment).
  // NameVertex*      _definition;

  //! Definitions of the name. Only relevant for use vertices.
  std::vector<NameVertex*> _definitions;

  //! Uses of the name in expressions. Only relevant for definition vertices.
  std::vector<NameVertex*> _uses;

};

} // namespace language
} // namespace ranally

#endif
