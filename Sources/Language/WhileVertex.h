#ifndef INCLUDED_RANALLY_WHILEVERTEX
#define INCLUDED_RANALLY_WHILEVERTEX

#include "ExpressionVertex.h"
#include "StatementVertex.h"



namespace ranally {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class WhileVertex: public StatementVertex
{

  friend class WhileVertexTest;

private:

  boost::shared_ptr<ExpressionVertex> _condition;

  StatementVertices _trueStatements;

  StatementVertices _falseStatements;

protected:

public:

  LOKI_DEFINE_VISITABLE()

                   WhileVertex         (
                        boost::shared_ptr<ExpressionVertex> const& condition,
                        StatementVertices const& trueStatements,
                        StatementVertices const& falseStatements);

  /* virtual */    ~WhileVertex        ();

  boost::shared_ptr<ExpressionVertex> const& condition() const;

  StatementVertices const& trueStatements() const;

  StatementVertices const& falseStatements() const;

};

} // namespace language
} // namespace ranally

#endif
