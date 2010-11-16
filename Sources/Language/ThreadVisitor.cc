#include <iostream>

#include "ThreadVisitor.h"
#include "Vertices.h"



namespace ranally {

ThreadVisitor::ThreadVisitor()

  : _lastVertex(0)

{
}



ThreadVisitor::~ThreadVisitor()
{
}



void ThreadVisitor::visitStatements(
  StatementVertices const& statements)
{
}



// void ThreadVisitor::visitExpressions(
//   ExpressionVertices const& expressions)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   AssignmentVertex& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   FunctionVertex& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   OperatorVertex& vertex)
// {
// }



void ThreadVisitor::Visit(
  SyntaxVertex&)
{
  assert(false);
}



void ThreadVisitor::Visit(
  ScriptVertex& vertex)
{
  _lastVertex = &vertex;
  visitStatements(vertex.statements());
  assert(_lastVertex);
  _lastVertex->setSuccessor(&vertex);
}



// void ThreadVisitor::Visit(
//   StringVertex& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NameVertex& vertex)
// {
// }
// 
// 
// 
// template<typename T>
// void ThreadVisitor::Visit(
//   NumberVertex<T>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<int8_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<int16_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<int32_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<int64_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<uint8_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<uint16_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<uint32_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<uint64_t>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<float>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   NumberVertex<double>& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   IfVertex& vertex)
// {
// }
// 
// 
// 
// void ThreadVisitor::Visit(
//   WhileVertex& vertex)
// {
// }

} // namespace ranally

