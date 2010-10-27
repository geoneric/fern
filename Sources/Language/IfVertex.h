#ifndef INCLUDED_RANALLY_IFVERTEX
#define INCLUDED_RANALLY_IFVERTEX

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

private:

  boost::shared_ptr<ranally::ExpressionVertex> _condition;

  StatementVertices _trueStatements;

  StatementVertices _falseStatements;

protected:

public:

  LOKI_DEFINE_VISITABLE()

                   IfVertex      (
                        boost::shared_ptr<ExpressionVertex> const& condition,
                        StatementVertices const& trueStatements,
                        StatementVertices const& falseStatements);

  /* virtual */    ~IfVertex     ();

  boost::shared_ptr<ranally::ExpressionVertex> const& condition() const;

  StatementVertices const& trueStatements() const;

  StatementVertices const& falseStatements() const;

};

} // namespace ranally

#endif
