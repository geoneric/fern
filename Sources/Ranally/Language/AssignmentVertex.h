#ifndef INCLUDED_RANALLY_LANGUAGE_ASSIGNMENTVERTEX
#define INCLUDED_RANALLY_LANGUAGE_ASSIGNMENTVERTEX

#include "Ranally/Language/ExpressionVertex.h"
#include "Ranally/Language/StatementVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AssignmentVertex: public StatementVertex
{

  friend class AssignmentVertexTest;

public:

  LOKI_DEFINE_VISITABLE()

                   AssignmentVertex    (ExpressionVertices const& targets,
                                        ExpressionVertices const& expressions);

  /* virtual */    ~AssignmentVertex   ();

  ExpressionVertices const& targets    () const;

  ExpressionVertices const& expressions() const;

protected:

private:

  ExpressionVertices _targets;

  ExpressionVertices _expressions;

};

} // namespace language
} // namespace ranally

#endif
