#ifndef INCLUDED_RANALLY_ASSIGNMENTVERTEX
#define INCLUDED_RANALLY_ASSIGNMENTVERTEX

#include <vector>
#include <boost/shared_ptr.hpp>

#include "ExpressionVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class AssignmentVertex: public SyntaxVertex
{

  friend class AssignmentVertexTest;

public:

  typedef std::vector<boost::shared_ptr<ranally::ExpressionVertex> >
    ExpressionVertices;

private:

  ExpressionVertices _targets;

  ExpressionVertices _expressions;

protected:

public:

                   AssignmentVertex    (ExpressionVertices const& targets,
                                        ExpressionVertices const& expressions);

  /* virtual */    ~AssignmentVertex   ();

};

} // namespace ranally

#endif
