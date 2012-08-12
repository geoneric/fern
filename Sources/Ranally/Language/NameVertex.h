#ifndef INCLUDED_RANALLY_LANGUAGE_NAMEVERTEX
#define INCLUDED_RANALLY_LANGUAGE_NAMEVERTEX

#include <boost/foreach.hpp>
#include "Ranally/Language/ExpressionVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class NameVertex:
  public ExpressionVertex
{

  friend class NameVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

                   NameVertex          (String const& name);

                   NameVertex          (int lineNr,
                                        int colId,
                                        String const& name);

                   ~NameVertex         ();

  void             addDefinition       (NameVertex* vertex);

  std::vector<NameVertex*> const& definitions() const;

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

typedef boost::shared_ptr<NameVertex> NameVertexPtr;

inline std::ostream& operator<<(
  std::ostream& stream,
  NameVertex const& vertex)
{
  stream << "name: " << vertex.name().encodeInUTF8() << "\n";

  if(!vertex.definitions().empty()) {
    stream << "definitions:\n";

    BOOST_FOREACH(NameVertex const* definition, vertex.definitions()) {
      stream << *definition;
    }
  }

  if(!vertex.uses().empty()) {
    stream << "uses:\n";

    BOOST_FOREACH(NameVertex const* use, vertex.uses()) {
      stream << *use;
    }
  }

  return stream;
}

} // namespace language
} // namespace ranally

#endif
