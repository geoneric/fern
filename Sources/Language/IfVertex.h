#ifndef INCLUDED_RANALLY_IFVERTEX
#define INCLUDED_RANALLY_IFVERTEX

#include <vector>
#include <boost/shared_ptr.hpp>

#include "ExpressionVertex.h"
#include "StatementVertex.h"



namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class IfVertex: public StatementVertex
{

  friend class IfVertexTest;

public:

  typedef std::vector<boost::shared_ptr<ranally::StatementVertex> >
    StatementVertices;

private:

  boost::shared_ptr<ranally::ExpressionVertex> _condition;

  StatementVertices _trueStatements;

  StatementVertices _falseStatements;

protected:

public:

                   IfVertex      (
                        boost::shared_ptr<ExpressionVertex> const& condition,
                        StatementVertices const& trueStatements,
                        StatementVertices const& falseStatements);

  /* virtual */    ~IfVertex     ();

};

} // namespace ranally

#endif
