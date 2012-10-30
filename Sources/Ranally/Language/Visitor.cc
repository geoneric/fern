#include "Ranally/Language/Visitor.h"
#include <boost/foreach.hpp>
#include "Ranally/Language/AssignmentVertex.h"
#include "Ranally/Language/ExpressionVertex.h"
#include "Ranally/Language/FunctionVertex.h"
#include "Ranally/Language/IfVertex.h"
#include "Ranally/Language/NameVertex.h"
#include "Ranally/Language/NumberVertex.h"
#include "Ranally/Language/OperationVertex.h"
#include "Ranally/Language/OperatorVertex.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/StatementVertex.h"
#include "Ranally/Language/StringVertex.h"
#include "Ranally/Language/WhileVertex.h"


namespace ranally {
namespace language {

//! Default construct an instance.
/*!
*/
Visitor::Visitor()
{
}


//! Destruct an instance.
/*!
*/
Visitor::~Visitor()
{
}


//! Allow all \a statements to accept the visitor.
/*!
  \param     statements Collection with StatementVertex instanceѕ.
*/
void Visitor::visitStatements(
    StatementVertices& statements)
{
    BOOST_FOREACH(boost::shared_ptr<StatementVertex> statementVertex,
        statements) {
        statementVertex->Accept(*this);
    }
}


//! Allow all \a expressions to accept the visitor.
/*!
  \param     expressions Collection with ExpressionVertex instanceѕ.
*/
void Visitor::visitExpressions(
  ExpressionVertices const& expressions)
{
    BOOST_FOREACH(boost::shared_ptr<ExpressionVertex> expressionVertex,
        expressions) {
        expressionVertex->Accept(*this);
    }
}


//! Visit an AssignmentVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation allows the source and target expressions
  to accept the visitor, in that order. After that it calls
  Visit(StatementVertex&).
*/
void Visitor::Visit(
    AssignmentVertex& vertex)
{
    vertex.expression()->Accept(*this);
    vertex.target()->Accept(*this);
    Visit(dynamic_cast<StatementVertex&>(vertex));
}


//! Visit an ExpressionVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(StatementVertex&);
*/
void Visitor::Visit(
    ExpressionVertex& vertex)
{
    Visit(dynamic_cast<StatementVertex&>(vertex));
}


//! Visit an FunctionVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(OperationVertex&);
*/
void Visitor::Visit(
    FunctionVertex& vertex)
{
    Visit(dynamic_cast<OperationVertex&>(vertex));
}


//! Visit an IfVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation allows the condition expression, and the true and
  false statements to accept the visitor, in that order. After that it calls
  Visit(StatementVertex&).
*/
void Visitor::Visit(
    IfVertex& vertex)
{
    vertex.condition()->Accept(*this);
    visitStatements(vertex.trueStatements());
    visitStatements(vertex.falseStatements());
    Visit(dynamic_cast<StatementVertex&>(vertex));
}


//! Visit a NameVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(ExpressionVertex&);
*/
void Visitor::Visit(
    NameVertex& vertex)
{
    Visit(dynamic_cast<ExpressionVertex&>(vertex));
}


//! Visit a NumberVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(ExpressionVertex&);
*/
#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void Visitor::Visit(                                                           \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    Visit(dynamic_cast<ExpressionVertex&>(vertex));                            \
}

VISIT_NUMBER_VERTEX(int8_t  )
VISIT_NUMBER_VERTEX(int16_t )
VISIT_NUMBER_VERTEX(int32_t )
VISIT_NUMBER_VERTEX(int64_t )
VISIT_NUMBER_VERTEX(uint8_t )
VISIT_NUMBER_VERTEX(uint16_t)
VISIT_NUMBER_VERTEX(uint32_t)
VISIT_NUMBER_VERTEX(uint64_t)
VISIT_NUMBER_VERTEX(float   )
VISIT_NUMBER_VERTEX(double  )

#undef VISIT_NUMBER_VERTEX


//! Visit an OperationVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation allows the argument expressions to accept the
  visitor. After that it calls Visit(ExpressionVertex&);
*/
void Visitor::Visit(
    OperationVertex& vertex)
{
    visitExpressions(vertex.expressions());
    Visit(dynamic_cast<ExpressionVertex&>(vertex));
}


//! Visit an OperatorVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(OperationVertex&);
*/
void Visitor::Visit(
    OperatorVertex& vertex)
{
    Visit(dynamic_cast<OperationVertex&>(vertex));
}


//! Visit a ScriptVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation allows the statements to accept the visitor.
  After that it calls Visit(SyntaxVertex&).
*/
void Visitor::Visit(
    ScriptVertex& vertex)
{
    visitStatements(vertex.statements());
    Visit(dynamic_cast<SyntaxVertex&>(vertex));
}


//! Visit a StatementVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(SyntaxVertex&);
*/
void Visitor::Visit(
    StatementVertex& vertex)
{
    Visit(dynamic_cast<SyntaxVertex&>(vertex));
}


//! Visit a StringVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(ExpressionVertex&);
*/
void Visitor::Visit(
    StringVertex& vertex)
{
    Visit(dynamic_cast<ExpressionVertex&>(vertex));
}


//! Visit a SyntaxVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation does nothing.
*/
void Visitor::Visit(
    SyntaxVertex& /* vertex */)
{
}


//! Visit a WhileVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation allows the condition expression, and the true and
  false statements to accept the visitor, in that order. After that it calls
  Visit(StatementVertex&).
*/
void Visitor::Visit(
    WhileVertex& vertex)
{
    vertex.condition()->Accept(*this);
    visitStatements(vertex.trueStatements());
    visitStatements(vertex.falseStatements());
    Visit(dynamic_cast<StatementVertex&>(vertex));
}

} // namespace language
} // namespace ranally
