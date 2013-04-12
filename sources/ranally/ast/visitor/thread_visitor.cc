#include "ranally/ast/visitor/thread_visitor.h"
#include "ranally/ast/core/vertices.h"


namespace ranally {

ThreadVisitor::ThreadVisitor()

    : Visitor(),
      _last_vertex(0)

{
}


void ThreadVisitor::Visit(
    AssignmentVertex& vertex)
{
    vertex.expression()->Accept(*this);

    vertex.target()->Accept(*this);

    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
    _last_vertex = &vertex;
}


void ThreadVisitor::Visit(
    FunctionVertex& vertex)
{
    visit_expressions(vertex.expressions());

    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
    _last_vertex = &vertex;
}


void ThreadVisitor::Visit(
    OperatorVertex& vertex)
{
    visit_expressions(vertex.expressions());

    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
    _last_vertex = &vertex;
}


void ThreadVisitor::Visit(
    SyntaxVertex&)
{
    assert(false);
}


void ThreadVisitor::Visit(
    ScriptVertex& vertex)
{
    _last_vertex = &vertex;
    visit_statements(vertex.statements());
    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
}


void ThreadVisitor::Visit(
    StringVertex& vertex)
{
    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
    _last_vertex = &vertex;
}


void ThreadVisitor::Visit(
    NameVertex& vertex)
{
    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
    _last_vertex = &vertex;
}


void ThreadVisitor::Visit(
    SubscriptVertex& vertex)
{
    // First we must get the control.
    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);

    // Let the main expression thread itself.
    _last_vertex = &vertex;
    vertex.expression()->Accept(*this);
    _last_vertex->add_successor(&vertex);

    // Let the selection thread itself.
    _last_vertex = &vertex;
    vertex.selection()->Accept(*this);
    _last_vertex->add_successor(&vertex);

    _last_vertex = &vertex;
}


template<typename T>
void ThreadVisitor::Visit(
    NumberVertex<T>& vertex)
{
    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
    _last_vertex = &vertex;
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
    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);

    // Let the true and false block thread themselves.
    _last_vertex = &vertex;
    assert(!vertex.true_statements().empty());
    visit_statements(vertex.true_statements());
    _last_vertex->add_successor(&vertex);

    if(!vertex.false_statements().empty()) {
        _last_vertex = &vertex;
        visit_statements(vertex.false_statements());
        _last_vertex->add_successor(&vertex);
    }

    _last_vertex = &vertex;
}


void ThreadVisitor::Visit(
    WhileVertex& /* vertex */)
{
    // TODO
    assert(false);
}

} // namespace ranally
