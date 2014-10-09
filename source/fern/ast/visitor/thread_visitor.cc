#include "fern/ast/visitor/thread_visitor.h"
#include "fern/ast/core/vertices.h"


namespace fern {

ThreadVisitor::ThreadVisitor()

    : AstVisitor(),
      _last_vertex(nullptr),
      _symbol_table(),
      _function_definitions(),
      _modes()

{
}


void ThreadVisitor::thread_to(
    AstVertex* vertex)
{
    if(_last_vertex) {
        _last_vertex->set_successor(vertex);
    }
    _last_vertex = vertex;
}


void ThreadVisitor::Visit(
    AssignmentVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            vertex.expression()->Accept(*this);
            vertex.target()->Accept(*this);
            assert(_last_vertex);
            thread_to(&vertex);
            break;
        }
    }
}


void ThreadVisitor::Visit(
    FunctionCallVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            visit_expressions(vertex.expressions());

            thread_to(&vertex);

            // If the function calls a user defined function, then we
            // must pass control to it. When threading the AST, a user
            // defined function takes precedence over a built-in function
            // with the same name.
            // The function definition is already threaded, so we can just
            // hook it up here.
            if(_symbol_table.has_value(vertex.name())) {
                FunctionDefinitionVertex* function_definition =
                    _symbol_table.value(vertex.name());

                // Entry point of definition.
                thread_to(function_definition);

                // Exit point of definition.
                _last_vertex = &(*function_definition->scope()->sentinel());
            }

            break;
        }
    }
}


void ThreadVisitor::Visit(
    FunctionDefinitionVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Don't hook up this definition to the current _last_vertex. It
            // is up to the caller to hook the definition up appropriately
            // (ѕee Visit(FunctionCallVertex)).
            AstVertex* original_last_vertex = _last_vertex;

            // Push the current function definition being processed. This is
            // used by return vertices to connect to the function's sentinel.
            _function_definitions.push(&vertex);

            _last_vertex = &vertex;

            _modes.push(Mode::VisitNonFunctionDefinitionStatements);
            visit_expressions(vertex.arguments());
            _modes.pop();

            Visit(*vertex.scope());
            assert(_last_vertex == &*vertex.scope()->sentinel());

            _function_definitions.pop();

            // Store the vertex to the symbol table of the current scope, so
            // we can connect function calls to this function's definition.
            _symbol_table.add_value(vertex.name(), &vertex);

            _last_vertex = original_last_vertex;
            assert(_last_vertex != &*vertex.scope()->sentinel());
            assert(!vertex.scope()->sentinel()->has_successor());
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            // Do nothing. Function definition is already threaded.
            break;
        }
    }
}


void ThreadVisitor::Visit(
    OperatorVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            visit_expressions(vertex.expressions());
            thread_to(&vertex);
            break;
        }
    }
}


void ThreadVisitor::Visit(
    AstVertex&)
{
    assert(false);
}


void ThreadVisitor::Visit(
    ModuleVertex& vertex)
{
    assert(_symbol_table.empty());
    assert(_function_definitions.empty());
    _last_vertex = &vertex;
    Visit(*vertex.scope());
    assert(_last_vertex);
    _last_vertex->set_successor(&vertex);
    assert(_function_definitions.empty());
    assert(_symbol_table.empty());
}


void ThreadVisitor::Visit(
    StringVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            thread_to(&vertex);
            break;
        }
    }
}


void ThreadVisitor::Visit(
    NameVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            thread_to(&vertex);
            break;
        }
    }
}


void ThreadVisitor::Visit(
    SubscriptVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            vertex.expression()->Accept(*this);
            vertex.selection()->Accept(*this);
            thread_to(&vertex);
            break;
        }
    }
}


void ThreadVisitor::Visit(
    AttributeVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            vertex.expression()->Accept(*this);
            thread_to(&vertex);
            break;
        }
    }
}


template<typename T>
void ThreadVisitor::Visit(
    NumberVertex<T>& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            thread_to(&vertex);
            break;
        }
    }
}


#define VISIT_NUMBER_VERTEX(                                                   \
    type)                                                                      \
void ThreadVisitor::Visit(                                                     \
    NumberVertex<type>& vertex)                                                \
{                                                                              \
    Visit<type>(vertex);                                                       \
}

VISIT_NUMBER_VERTICES(VISIT_NUMBER_VERTEX)

#undef VISIT_NUMBER_VERTEX


void ThreadVisitor::Visit(
    ReturnVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            if(vertex.expression()) {
                vertex.expression()->Accept(*this);
            }
            thread_to(&vertex);

            // We can't assume that the next vertex will be the sentinel. It
            // is allowed to put stuff after a return statement. So we need
            // to explicitly connect to the sentinel ourselves. See
            // Visit(ScopeVertex) for more info.
            _last_vertex->set_successor(
                &*_function_definitions.top()->scope()->sentinel());
            break;
        }
    }
}


void ThreadVisitor::Visit(
    SentinelVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            assert(_last_vertex);
            _last_vertex->set_successor(&vertex);
            _last_vertex = &vertex;
            break;
        }
    }
}


void ThreadVisitor::Visit(
    IfVertex& vertex)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            // First let the condition thread itself.
            vertex.condition()->Accept(*this);

            // Now we must get the control.
            assert(_last_vertex);
            _last_vertex->set_successor(&vertex);

            // Let the true and false block thread themselves.
            _last_vertex = &vertex;
            Visit(*vertex.true_scope());
            assert(_last_vertex == &(*vertex.true_scope()->sentinel()));
            _last_vertex->set_successor(&(*vertex.sentinel()));

            if(!vertex.false_scope()->statements().empty()) {
                _last_vertex = &vertex;
                Visit(*vertex.false_scope());
                assert(_last_vertex == &(*vertex.false_scope()->sentinel()));
                _last_vertex->set_successor(&(*vertex.sentinel()));
            }

            _last_vertex = &(*vertex.sentinel());
            break;
        }
    }
}


void ThreadVisitor::Visit(
    WhileVertex& /* vertex */)
{
    switch(_modes.top()) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Do nothing. This is not a function definition.
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            break;
        }
    }
    // TODO
    assert(false);
}


void ThreadVisitor::Visit(
    ScopeVertex& vertex)
{
    // Whenever a scope is threaded, a first pass must be performed to
    // thread the user defined functions that may be present. After that the
    // other statements are threaded and any function calls to user-defined
    // functions are connected to the function definition.

    _symbol_table.push_scope();

    // Let the function definitions thread themselves.
    // Store the current last_vertex, so we can restore it later. The
    // function definition's statements are threaded now, but aren't
    // connected to the current thread. Only when the function is called
    // is the function definition's thread connected to the main thread
    // (see Visit(FunctionCallVertex&)).
    {
        assert(_last_vertex);
        AstVertex* original_last_vertex = _last_vertex;

        _modes.push(Mode::VisitFunctionDefinitionStatements);
        _last_vertex = &vertex;
        visit_statements(vertex.statements());
        Visit(*vertex.sentinel());
        _last_vertex = original_last_vertex;
        _modes.pop();
    }

    // Thread all statements, except for the function definitions. When a
    // function call is encountered, threading connects the call site with
    // the definition's entry point.
    {
        _modes.push(Mode::VisitNonFunctionDefinitionStatements);

        // The only reason we call add_successor instead of set_successor is
        // that when _last_vertex is an IfVertex, its true and false scopes are
        // both added as successors. Most vertices have exactly one successor.
        assert(_last_vertex);
        _last_vertex->add_successor(&vertex);
        _last_vertex = &vertex;

        for(auto statementVertex: vertex.statements()) {
            statementVertex->Accept(*this);
            assert(_last_vertex);
            // Iff the last vertex' successor is the sentinel of this scope,
            // then the last statement was the return statement, which
            // explicitly connected to the sentinel. In that case we reset
            // the _last_vertex pointer to null. Subsequent statements are
            // threaded but they are not connected to the control flow graph.
            // This is the reason that leave vertices (expressions,
            // return statement) have to test whether _last_vertex is not
            // null before attaching (they call thread_to(AstVertex)).
            if(_last_vertex->has_successor()) {
                assert(_last_vertex->successor() == &*vertex.sentinel());
                _last_vertex = nullptr;
            }
        }

        // If _last_vertex is the null vertex, then the last statement was
        // a return statement which is already connected to the sentinel
        // vertex. Otherwise, just visit the sentinel vertex.
        if(!_last_vertex) {
            assert(vertex.statements().back()->successor() ==
                &*vertex.sentinel());
            _last_vertex = &*vertex.sentinel();
        }
        else {
            Visit(*vertex.sentinel());
        }

        _modes.pop();
    }

    _symbol_table.pop_scope();
}

} // namespace fern
