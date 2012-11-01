#include "Ranally/Language/ThreadVisitor.h"
#include "Ranally/Language/Vertices.h"


namespace ranally {

ThreadVisitor::ThreadVisitor()

    : Visitor(),
      _lastVertex(0)

{
}


ThreadVisitor::~ThreadVisitor()
{
}


void ThreadVisitor::Visit(
    AssignmentVertex& vertex)
{
    vertex.expression()->Accept(*this);

    vertex.target()->Accept(*this);

    assert(_lastVertex);
    _lastVertex->addSuccessor(&vertex);
    _lastVertex = &vertex;
}


void ThreadVisitor::Visit(
    FunctionVertex& vertex)
{
    visitExpressions(vertex.expressions());

    assert(_lastVertex);
    _lastVertex->addSuccessor(&vertex);
    _lastVertex = &vertex;
}


void ThreadVisitor::Visit(
    OperatorVertex& vertex)
{
    visitExpressions(vertex.expressions());

    assert(_lastVertex);
    _lastVertex->addSuccessor(&vertex);
    _lastVertex = &vertex;
}


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
    _lastVertex->addSuccessor(&vertex);
}


void ThreadVisitor::Visit(
    StringVertex& vertex)
{
    assert(_lastVertex);
    _lastVertex->addSuccessor(&vertex);
    _lastVertex = &vertex;
}


void ThreadVisitor::Visit(
    NameVertex& vertex)
{
    assert(_lastVertex);
    _lastVertex->addSuccessor(&vertex);
    _lastVertex = &vertex;
}


template<typename T>
void ThreadVisitor::Visit(
    NumberVertex<T>& vertex)
{
    assert(_lastVertex);
    _lastVertex->addSuccessor(&vertex);
    _lastVertex = &vertex;
}


void ThreadVisitor::Visit(
    NumberVertex<int8_t>& vertex)
{
    Visit<int8_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<int16_t>& vertex)
{
    Visit<int16_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<int32_t>& vertex)
{
    Visit<int32_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<int64_t>& vertex)
{
    Visit<int64_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<uint8_t>& vertex)
{
    Visit<uint8_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<uint16_t>& vertex)
{
    Visit<uint16_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<uint32_t>& vertex)
{
    Visit<uint32_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<uint64_t>& vertex)
{
    Visit<uint64_t>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<float>& vertex)
{
    Visit<float>(vertex);
}


void ThreadVisitor::Visit(
    NumberVertex<double>& vertex)
{
    Visit<double>(vertex);
}


void ThreadVisitor::Visit(
    IfVertex& vertex)
{
    // First let the condition thread itself.
    vertex.condition()->Accept(*this);

    // Now we must get the control.
    assert(_lastVertex);
    _lastVertex->addSuccessor(&vertex);

    // Let the true and false block thread themselves.
    _lastVertex = &vertex;
    assert(!vertex.trueStatements().empty());
    visitStatements(vertex.trueStatements());
    _lastVertex->addSuccessor(&vertex);

    if(!vertex.falseStatements().empty()) {
        _lastVertex = &vertex;
        visitStatements(vertex.falseStatements());
        _lastVertex->addSuccessor(&vertex);
    }

    _lastVertex = &vertex;
}


void ThreadVisitor::Visit(
    WhileVertex& /* vertex */)
{
    // TODO
    assert(false);
}

} // namespace ranally
