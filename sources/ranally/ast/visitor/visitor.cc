#include "ranally/ast/visitor/visitor.h"
#include "ranally/ast/core/vertices.h"


namespace ranally {

//! Allow all \a statements to accept the visitor.
/*!
  \param     statements Collection with StatementVertex instanceѕ.
*/
void Visitor::visit_statements(
    StatementVertices& statements)
{
    for(auto statementVertex: statements) {
        statementVertex->Accept(*this);
    }
}


//! Allow all \a expressions to accept the visitor.
/*!
  \param     expressions Collection with ExpressionVertex instanceѕ.
*/
void Visitor::visit_expressions(
  ExpressionVertices const& expressions)
{
    for(auto expressionVertex: expressions) {
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


//! Visit a FunctionDefinitionVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation allows the the argument expressions, the
  block statements and the sentinel to accept the visitor, in that order.
  After that it calls Visit(StatementVertex&).
*/
void Visitor::Visit(
    FunctionDefinitionVertex& vertex)
{
    visit_expressions(vertex.arguments());
    Visit(*vertex.scope());
    Visit(dynamic_cast<StatementVertex&>(vertex));
}


//! Visit a FunctionCallVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(OperationVertex&);
*/
void Visitor::Visit(
    FunctionCallVertex& vertex)
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
    Visit(*vertex.true_scope());
    Visit(*vertex.false_scope());
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

VISIT_NUMBER_VERTICES(VISIT_NUMBER_VERTEX)

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
    visit_expressions(vertex.expressions());
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


//! Visit a ReturnVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(StatementVertex&);

  The default implementation visits the expression. After that it calls
  Visit(StatementVertex&).
*/
void Visitor::Visit(
    ReturnVertex& vertex)
{
    if(vertex.expression()) {
        vertex.expression()->Accept(*this);
    }
    Visit(dynamic_cast<StatementVertex&>(vertex));
}


//! Visit a ModuleVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation allows the statements to accept the visitor.
  After that it calls Visit(AstVertex&).
*/
void Visitor::Visit(
    ModuleVertex& vertex)
{
    Visit(*vertex.scope());
    Visit(dynamic_cast<AstVertex&>(vertex));
}


//! Visit a StatementVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation calls Visit(AstVertex&);
*/
void Visitor::Visit(
    StatementVertex& vertex)
{
    Visit(dynamic_cast<AstVertex&>(vertex));
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


//! Visit a SubscriptVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation visits the expression and the selection. After
  that it calls Visit(ExpressionVertex&).
*/
void Visitor::Visit(
    SubscriptVertex& vertex)
{
    vertex.expression()->Accept(*this);
    vertex.selection()->Accept(*this);
    Visit(dynamic_cast<ExpressionVertex&>(vertex));
}


//! Visit a AstVertex instance.
/*!
  \param     vertex Vertex to visit.

  The default implementation does nothing.
*/
void Visitor::Visit(
    AstVertex& /* vertex */)
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
    Visit(*vertex.true_scope());
    Visit(*vertex.false_scope());
    Visit(dynamic_cast<StatementVertex&>(vertex));
}


void Visitor::Visit(
    SentinelVertex& vertex)
{
    Visit(dynamic_cast<AstVertex&>(vertex));
}


void Visitor::Visit(
    ScopeVertex& vertex)
{
    visit_statements(vertex.statements());
    Visit(*vertex.sentinel());
    Visit(dynamic_cast<AstVertex&>(vertex));
}

} // namespace ranally
