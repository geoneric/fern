#include "ranally/ast/visitor/thread_visitor.h"
#include "ranally/ast/core/vertices.h"


namespace ranally {

ThreadVisitor::ThreadVisitor()

    : Visitor(),
      _last_vertex(nullptr),
      _symbol_table(),
      _function_definitions(),
      _mode(Mode::VisitFunctionDefinitionStatements)

{
}


void ThreadVisitor::Visit(
    AssignmentVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            vertex.expression()->Accept(*this);

            vertex.target()->Accept(*this);

            assert(_last_vertex);
            _last_vertex->add_successor(&vertex);
            _last_vertex = &vertex;
            break;
        }
    }
}


void ThreadVisitor::Visit(
    FunctionVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            visit_expressions(vertex.expressions());

            assert(_last_vertex);
            _last_vertex->add_successor(&vertex);
            _last_vertex = &vertex;

            // If the function calls a user defined function, then we
            // must pass control to it. When threading the AST, a user
            // defined function takes precedence over a built-in function
            // with the same name.
            // The function definition is already threaded, so we can just
            // hook it up here.
            if(_symbol_table.has_value(vertex.name())) {
                FunctionDefinitionVertex* function_definition =
                    _symbol_table.value(vertex.name());
                _last_vertex->add_successor(function_definition);
                // TODO Add an exit vertex to the AST. All return statements
                //      must end there. Also the last statement of the function
                //      body must end there.
                // hier verder
                // _last_vertex = &(*function_definition->exit_vertex());
            }

            break;
        }
    }
}


void ThreadVisitor::Visit(
    FunctionDefinitionVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            // Store the current last_vertex, so we can restore it later. The
            // function definition's statements are threaded now, but aren't
            // connected to the current thread. Only when the function is called
            // is the function definition's thread connected to the main thread
            // (see Visit(FunctionVertex&)).
            SyntaxVertex*  original_last_vertex = _last_vertex;
            _last_vertex = &vertex;

            // Push the current function definition being processed. This is
            // used by return vertices.
            _function_definitions.push(&vertex);

            visit_scope(vertex.body());
            _function_definitions.pop();
            _last_vertex = original_last_vertex;

            // Store the vertex to the symbol table of the current scope, so
            // we can connect function calls to this function's definition.
            _symbol_table.add_value(vertex.name(), &vertex);
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            // TODO Don't know for sure that we a being called...

            // // We are being called. Connect thread from function call to
            // // function definition.
            // _last_vertex->add_successor(&vertex);
            // // TODO The FunctionDefinitionVertex is the entry point of the
            // //      function definition. Add an exit point where we can
            // //      connect return statements to. This exit point is also
            // //      the vertex we want to assign here:
            // _last_vertex = &vertex;
            break;
        }
    }




    // The argument expressions are assigned the values that are passed in.
    // _last_vertex is pointing to the function call. This function call
    // has the argument expressions that need to be connected to the argument
    // expressions of the function definitions.

    // FunctionVertex* function_vertex = dynamic_cast<FunctionVertex*>(
    //     _last_vertex);
    // assert(function_vertex);

    // for(size_t i = 0; i < function_vertex->expressions().size(); ++i) {
    //     if(i < vertex.arguments().size()) {
    //         function_vertex->expressions()[i]->add_successor(
    //             &(*vertex.arguments()[i]));
    //     }
    // }

    // _last_vertex->add_successor(&vertex);

    // visit_expressions(vertex.arguments());
    // visit_statements(vertex.body());

    // _last_vertex = &vertex;
}


void ThreadVisitor::Visit(
    OperatorVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            visit_expressions(vertex.expressions());

            assert(_last_vertex);
            _last_vertex->add_successor(&vertex);
            _last_vertex = &vertex;
            break;
        }
    }
}


void ThreadVisitor::Visit(
    SyntaxVertex&)
{
    assert(false);
}


void ThreadVisitor::Visit(
    ScriptVertex& vertex)
{
    assert(_symbol_table.empty());
    assert(_function_definitions.empty());
    _last_vertex = &vertex;
    visit_scope(vertex.statements());
    assert(_last_vertex);
    _last_vertex->add_successor(&vertex);
    assert(_function_definitions.empty());
    assert(_symbol_table.empty());
}


void ThreadVisitor::Visit(
    StringVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            assert(_last_vertex);
            _last_vertex->add_successor(&vertex);
            _last_vertex = &vertex;
            break;
        }
    }
}


void ThreadVisitor::Visit(
    NameVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            assert(_last_vertex);
            _last_vertex->add_successor(&vertex);
            _last_vertex = &vertex;
            break;
        }
    }
}


void ThreadVisitor::Visit(
    SubscriptVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
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
            break;
        }
    }
}


template<typename T>
void ThreadVisitor::Visit(
    NumberVertex<T>& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            assert(_last_vertex);
            _last_vertex->add_successor(&vertex);
            _last_vertex = &vertex;
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
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            if(vertex.expression()) {
                vertex.expression()->Accept(*this);
            }
            _last_vertex->add_successor(&vertex);
            assert(!_function_definitions.empty());
            vertex.add_successor(_function_definitions.top());
            break;
        }
    }
}


void ThreadVisitor::Visit(
    IfVertex& vertex)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
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
            break;
        }
    }
}


void ThreadVisitor::Visit(
    WhileVertex& /* vertex */)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            break;
        }
    }
    // TODO
    assert(false);
}


void ThreadVisitor::visit_scope(
    StatementVertices& statements)
{
    switch(_mode) {
        case Mode::VisitFunctionDefinitionStatements: {
            break;
        }
        case Mode::VisitNonFunctionDefinitionStatements: {
            break;
        }
    }

    // Pass 1: Visit function definitions.
    // Pass 2. Visit all other statements.
    _symbol_table.push_scope();

    // Let the function definitions thread themselves.
    _mode = Mode::VisitFunctionDefinitionStatements;
    visit_statements(statements);

    // Thread all statements, except for the function definitions. When a
    // function call is encountered, threading connects the call site with
    // the definition's entry point, and the return statement(s) (if any),
    // with the identifier assigned to (if any).
    _mode = Mode::VisitNonFunctionDefinitionStatements;
    visit_statements(statements);

    _symbol_table.pop_scope();
}

} // namespace ranally
