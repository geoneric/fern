#ifndef INCLUDED_RANALLY_ASSIGNMENTVERTEX
#define INCLUDED_RANALLY_ASSIGNMENTVERTEX

#include "ExpressionVertex.h"
#include "StatementVertex.h"



namespace ranally {

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

private:

  ExpressionVertices _targets;

  ExpressionVertices _expressions;

protected:

public:

                   AssignmentVertex    (ExpressionVertices const& targets,
                                        ExpressionVertices const& expressions);

  /* virtual */    ~AssignmentVertex   ();

  ExpressionVertices const& targets    () const;

  ExpressionVertices const& expressions() const;

};

} // namespace ranally

#endif
