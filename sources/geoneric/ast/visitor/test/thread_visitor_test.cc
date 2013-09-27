#define BOOST_TEST_MODULE geoneric ast
#include <boost/test/unit_test.hpp>
#include "geoneric/core/string.h"
#include "geoneric/script/algebra_parser.h"
#include "geoneric/ast/core/vertices.h"
#include "geoneric/ast/visitor/thread_visitor.h"
#include "geoneric/ast/xml/xml_parser.h"

// #include <typeinfo>
// std::cout << typeid(*tree->scope()->successor()).name() << std::endl;


namespace boost {

template<class T1, class T2>
bool operator==(
        T1 const* lhs,
        std::shared_ptr<T2> const& rhs)
{
    return lhs == &(*rhs);
}

} // namespace boost


class Support
{

public:

    Support()
        : _algebra_parser(),
          _xml_parser(),
          _thread_visitor()
    {
    }

protected:

    geoneric::AlgebraParser _algebra_parser;

    geoneric::XmlParser _xml_parser;

    geoneric::ThreadVisitor _thread_visitor;

};


BOOST_FIXTURE_TEST_SUITE(thread_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("")));
    tree->Accept(_thread_visitor);
    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("a")));
    tree->Accept(_thread_visitor);

    geoneric::AstVertex const* vertex_a = &(*tree->scope()->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("a = b")));
    tree->Accept(_thread_visitor);

    geoneric::AssignmentVertex const* assignment =
        dynamic_cast<geoneric::AssignmentVertex const*>(
            &(*tree->scope()->statements()[0]));
    geoneric::AstVertex const* vertex_a = &(*assignment->target());
    geoneric::AstVertex const* vertex_b = &(*assignment->expression());

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), assignment);
    BOOST_CHECK_EQUAL(assignment->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("\"five\"")));
    tree->Accept(_thread_visitor);

    geoneric::AstVertex const* string_vertex =
        &(*tree->scope()->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), string_vertex);
    BOOST_CHECK_EQUAL(string_vertex->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("5")));
    tree->Accept(_thread_visitor);

    geoneric::AstVertex const* number_vertex =
        &(*tree->scope()->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), number_vertex);
    BOOST_CHECK_EQUAL(number_vertex->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("f()")));
        tree->Accept(_thread_visitor);

        geoneric::AstVertex const* function_call_vertex =
            &(*tree->scope()->statements()[0]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), function_call_vertex);
        BOOST_CHECK_EQUAL(function_call_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("f(1, \"2\", three, four())")));
        tree->Accept(_thread_visitor);

        geoneric::FunctionCallVertex const* function_call_vertex =
            dynamic_cast<geoneric::FunctionCallVertex const*>(
                &(*tree->scope()->statements()[0]));
        geoneric::AstVertex const* vertex1 =
            &(*function_call_vertex->expressions()[0]);
        geoneric::AstVertex const* vertex2 =
            &(*function_call_vertex->expressions()[1]);
        geoneric::AstVertex const* vertex3 =
            &(*function_call_vertex->expressions()[2]);
        geoneric::AstVertex const* vertex4 =
            &(*function_call_vertex->expressions()[3]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), vertex3);
        BOOST_CHECK_EQUAL(vertex3->successor(), vertex4);
        BOOST_CHECK_EQUAL(vertex4->successor(), function_call_vertex);
        BOOST_CHECK_EQUAL(function_call_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("-a")));
        tree->Accept(_thread_visitor);

        geoneric::OperatorVertex const* operator_vertex =
            dynamic_cast<geoneric::OperatorVertex const*>(
                &(*tree->scope()->statements()[0]));
        geoneric::AstVertex const* vertex1 =
            &(*operator_vertex->expressions()[0]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), operator_vertex);
        BOOST_CHECK_EQUAL(operator_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("a + b")));
        tree->Accept(_thread_visitor);

        geoneric::OperatorVertex const* operator_vertex =
            dynamic_cast<geoneric::OperatorVertex const*>(
                &(*tree->scope()->statements()[0]));
        geoneric::AstVertex const* vertex1 =
            &(*operator_vertex->expressions()[0]);
        geoneric::AstVertex const* vertex2 =
            &(*operator_vertex->expressions()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), operator_vertex);
        BOOST_CHECK_EQUAL(operator_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("-(a + b)")));
        tree->Accept(_thread_visitor);

        geoneric::OperatorVertex const* operator1_vertex =
            dynamic_cast<geoneric::OperatorVertex const*>(
                &(*tree->scope()->statements()[0]));
        geoneric::OperatorVertex const* operator2_vertex =
            dynamic_cast<geoneric::OperatorVertex const*>(
                &(*operator1_vertex->expressions()[0]));
        geoneric::AstVertex const* vertex1 =
            &(*operator2_vertex->expressions()[0]);
        geoneric::AstVertex const* vertex2 =
            &(*operator2_vertex->expressions()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), operator2_vertex);
        BOOST_CHECK_EQUAL(operator2_vertex->successor(), operator1_vertex);
        BOOST_CHECK_EQUAL(operator1_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }
}


BOOST_AUTO_TEST_CASE(visit_multiple_statement)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("a;b;c")));
    tree->Accept(_thread_visitor);

    geoneric::AstVertex const* vertex_a = &(*tree->scope()->statements()[0]);
    geoneric::AstVertex const* vertex_b = &(*tree->scope()->statements()[1]);
    geoneric::AstVertex const* vertex_c = &(*tree->scope()->statements()[2]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
    BOOST_CHECK_EQUAL(vertex_c->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_nested_expressions)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        geoneric::String("a = b + c")));
    tree->Accept(_thread_visitor);

    geoneric::AssignmentVertex const* assignment =
        dynamic_cast<geoneric::AssignmentVertex const*>(
            &(*tree->scope()->statements()[0]));
    geoneric::AstVertex const* vertex_a = &(*assignment->target());
    geoneric::OperatorVertex const* addition =
        dynamic_cast<geoneric::OperatorVertex const*>(
            &(*assignment->expression()));
    geoneric::AstVertex const* vertex_b =
        &(*addition->expressions()[0]);
    geoneric::AstVertex const* vertex_c =
        &(*addition->expressions()[1]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
    BOOST_CHECK_EQUAL(vertex_c->successor(), addition);
    BOOST_CHECK_EQUAL(addition->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), assignment);
    BOOST_CHECK_EQUAL(assignment->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(
                "if a:\n"
                "    b\n"
                "    c")));
        tree->Accept(_thread_visitor);

        geoneric::IfVertex const* if_vertex =
            dynamic_cast<geoneric::IfVertex const*>(
                &(*tree->scope()->statements()[0]));
        geoneric::AstVertex const* vertex_a =
            &(*if_vertex->condition());
        geoneric::AstVertex const* vertex_b =
            &(*if_vertex->true_scope()->statements()[0]);
        geoneric::AstVertex const* vertex_c =
            &(*if_vertex->true_scope()->statements()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_a);
        BOOST_CHECK_EQUAL(vertex_a->successor(), if_vertex);
        BOOST_CHECK_EQUAL(if_vertex->successor(0), if_vertex->true_scope());
        BOOST_CHECK_EQUAL(if_vertex->true_scope()->successor(), vertex_b);
        BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
        BOOST_CHECK_EQUAL(vertex_c->successor(),
            if_vertex->true_scope()->sentinel());
        BOOST_CHECK_EQUAL(if_vertex->true_scope()->sentinel()->successor(),
            if_vertex->sentinel());
        BOOST_CHECK_EQUAL(if_vertex->sentinel()->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(
                "if a:\n"
                "    b\n"
                "    c\n"
                "elif d:\n"
                "    e\n"
                "    f\n"
                "else:\n"
                "    g\n"
                "    h\n")));
        tree->Accept(_thread_visitor);

        // True block first if.
        geoneric::IfVertex const* if1_vertex =
            dynamic_cast<geoneric::IfVertex const*>(
                &(*tree->scope()->statements()[0]));
        geoneric::AstVertex const* vertex_a =
            &(*if1_vertex->condition());
        geoneric::AstVertex const* vertex_b =
            &(*if1_vertex->true_scope()->statements()[0]);
        geoneric::AstVertex const* vertex_c =
            &(*if1_vertex->true_scope()->statements()[1]);

        // True block second if.
        geoneric::IfVertex const* if2_vertex =
            dynamic_cast<geoneric::IfVertex const*>(
                &(*if1_vertex->false_scope()->statements()[0]));
        geoneric::AstVertex const* vertex_d =
            &(*if2_vertex->condition());
        geoneric::AstVertex const* vertex_e =
            &(*if2_vertex->true_scope()->statements()[0]);
        geoneric::AstVertex const* vertex_f =
            &(*if2_vertex->true_scope()->statements()[1]);

        // False block second if.
        geoneric::AstVertex const* vertex_g =
            &(*if2_vertex->false_scope()->statements()[0]);
        geoneric::AstVertex const* vertex_h =
            &(*if2_vertex->false_scope()->statements()[1]);

        // True block first if.
        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_a);
        BOOST_CHECK_EQUAL(vertex_a->successor(), if1_vertex);
        BOOST_CHECK_EQUAL(if1_vertex->successor(0), if1_vertex->true_scope());
        BOOST_CHECK_EQUAL(if1_vertex->true_scope()->successor(), vertex_b);
        BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
        BOOST_CHECK_EQUAL(vertex_c->successor(),
            if1_vertex->true_scope()->sentinel());
        BOOST_CHECK_EQUAL(if1_vertex->true_scope()->sentinel()->successor(),
            if1_vertex->sentinel());

        // True block second if.
        BOOST_CHECK_EQUAL(if1_vertex->successor(1), if1_vertex->false_scope());
        BOOST_CHECK_EQUAL(if1_vertex->false_scope()->successor(), vertex_d);
        BOOST_CHECK_EQUAL(vertex_d->successor(), if2_vertex);
        BOOST_CHECK_EQUAL(if2_vertex->successor(0), if2_vertex->true_scope());
        BOOST_CHECK_EQUAL(if2_vertex->true_scope()->successor(), vertex_e);
        BOOST_CHECK_EQUAL(vertex_e->successor(), vertex_f);
        BOOST_CHECK_EQUAL(vertex_f->successor(),
            if2_vertex->true_scope()->sentinel());
        BOOST_CHECK_EQUAL(if2_vertex->true_scope()->sentinel()->successor(),
            if2_vertex->sentinel());
        BOOST_CHECK_EQUAL(if2_vertex->sentinel()->successor(),
            if1_vertex->false_scope()->sentinel());
        BOOST_CHECK_EQUAL(if1_vertex->false_scope()->sentinel()->successor(),
            if1_vertex->sentinel());

        // False block second if.
        BOOST_CHECK_EQUAL(if2_vertex->successor(1), if2_vertex->false_scope());
        BOOST_CHECK_EQUAL(if2_vertex->false_scope()->successor(), vertex_g);
        BOOST_CHECK_EQUAL(vertex_g->successor(), vertex_h);
        BOOST_CHECK_EQUAL(vertex_h->successor(),
            if2_vertex->false_scope()->sentinel());
        BOOST_CHECK_EQUAL(if2_vertex->false_scope()->sentinel()->successor(),
            if2_vertex->sentinel());
        BOOST_CHECK_EQUAL(if2_vertex->sentinel()->successor(),
            if1_vertex->false_scope()->sentinel());
        BOOST_CHECK_EQUAL(if1_vertex->false_scope()->sentinel()->successor(),
            if1_vertex->sentinel());

        BOOST_CHECK_EQUAL(if1_vertex->sentinel()->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }
}


BOOST_AUTO_TEST_CASE(visit_while)
{
}


BOOST_AUTO_TEST_CASE(visit_function_definition)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
def foo():
    return
)")));
        tree->Accept(_thread_visitor);

        // The defined function isn't called, so the script is in effect
        // empty. This doesn't mean the definition isn't in the script. It
        // means threading hasn't connected to the function.
        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        geoneric::FunctionDefinitionVertex const* function_definition_vertex =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(function_definition_vertex);

        BOOST_REQUIRE_EQUAL(
            function_definition_vertex->scope()->statements().size(), 1u);
        geoneric::ReturnVertex const* return_vertex =
            dynamic_cast<geoneric::ReturnVertex const*>(
                &(*function_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);
        BOOST_CHECK(!return_vertex->expression());

        BOOST_CHECK_EQUAL(function_definition_vertex->successor(),
            function_definition_vertex->scope());
        BOOST_CHECK_EQUAL(function_definition_vertex->scope()->successor(),
            return_vertex);
        BOOST_CHECK_EQUAL(return_vertex->successor(),
            function_definition_vertex->scope()->sentinel());
        BOOST_CHECK(
            !function_definition_vertex->scope()->sentinel()->has_successor());
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
def foo():
    return 5
)")));
        tree->Accept(_thread_visitor);

        // The defined function isn't called, so the script is in effect
        // empty. This doesn't mean the definition isn't in the script. It
        // means threading hasn't connected to the function.
        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        geoneric::FunctionDefinitionVertex const* vertex_foo =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(vertex_foo);

        BOOST_REQUIRE_EQUAL(vertex_foo->scope()->statements().size(), 1u);
        geoneric::ReturnVertex const* return_vertex =
            dynamic_cast<geoneric::ReturnVertex const*>(
                &(*vertex_foo->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);

        BOOST_REQUIRE(return_vertex->expression());
        geoneric::AstVertex const* number_vertex =
            &(*return_vertex->expression());

        BOOST_CHECK_EQUAL(vertex_foo->successor(), vertex_foo->scope());
        BOOST_CHECK_EQUAL(vertex_foo->scope()->successor(), number_vertex);
        BOOST_CHECK_EQUAL(number_vertex->successor(), return_vertex);
        BOOST_CHECK_EQUAL(return_vertex->successor(),
            vertex_foo->scope()->sentinel());
        BOOST_CHECK(!vertex_foo->scope()->sentinel()->has_successor());
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
def foo():
    return 5

a = foo())")));
        tree->Accept(_thread_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);

        geoneric::FunctionDefinitionVertex const* vertex_foo =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(vertex_foo);

        BOOST_REQUIRE_EQUAL(vertex_foo->scope()->statements().size(), 1u);
        geoneric::ReturnVertex const* return_vertex =
            dynamic_cast<geoneric::ReturnVertex const*>(
                &(*vertex_foo->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);

        BOOST_REQUIRE(return_vertex->expression());
        geoneric::NumberVertex<int64_t> const* number_vertex =
            dynamic_cast<geoneric::NumberVertex<int64_t> const*>(
                &(*return_vertex->expression()));

        geoneric::AssignmentVertex const* assignment_vertex =
            dynamic_cast<geoneric::AssignmentVertex const*>(
                &(*tree->scope()->statements()[1]));
        BOOST_REQUIRE(assignment_vertex);

        geoneric::FunctionCallVertex const* function_call_vertex =
            dynamic_cast<geoneric::FunctionCallVertex const*>(
                &(*assignment_vertex->expression()));
        BOOST_REQUIRE(function_call_vertex);

        geoneric::NameVertex const* name_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*assignment_vertex->target()));
        BOOST_REQUIRE(name_vertex);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), function_call_vertex);
        BOOST_CHECK_EQUAL(function_call_vertex->successor(), vertex_foo);
        BOOST_CHECK_EQUAL(vertex_foo->successor(), vertex_foo->scope());
        BOOST_CHECK_EQUAL(vertex_foo->scope()->successor(), number_vertex);
        BOOST_CHECK_EQUAL(number_vertex->successor(), return_vertex);
        BOOST_CHECK_EQUAL(return_vertex->successor(),
            vertex_foo->scope()->sentinel());
        BOOST_REQUIRE(vertex_foo->scope()->sentinel()->has_successor());
        BOOST_CHECK_EQUAL(vertex_foo->scope()->sentinel()->successor(),
            name_vertex);
        BOOST_CHECK_EQUAL(name_vertex->successor(), assignment_vertex);
        BOOST_CHECK_EQUAL(assignment_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        // Test function without return statement.
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
def foo():
    bar

foo())")));
        tree->Accept(_thread_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);

        geoneric::FunctionDefinitionVertex const* function_definition_vertex =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(function_definition_vertex);

        BOOST_REQUIRE_EQUAL(
            function_definition_vertex->scope()->statements().size(), 1u);
        geoneric::NameVertex const* name_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*function_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(name_vertex);

        geoneric::FunctionCallVertex const* function_call_vertex =
            dynamic_cast<geoneric::FunctionCallVertex const*>(
                &(*tree->scope()->statements()[1]));
        BOOST_REQUIRE(function_call_vertex);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), function_call_vertex);
        BOOST_CHECK_EQUAL(function_call_vertex->successor(),
            function_definition_vertex);
        BOOST_CHECK_EQUAL(function_definition_vertex->successor(),
            function_definition_vertex->scope());
        BOOST_CHECK_EQUAL(function_definition_vertex->scope()->successor(),
            name_vertex);
        BOOST_CHECK_EQUAL(name_vertex->successor(), 
            function_definition_vertex->scope()->sentinel());
        BOOST_REQUIRE(
            function_definition_vertex->scope()->sentinel()->has_successor());
        BOOST_CHECK_EQUAL(
            function_definition_vertex->scope()->sentinel()->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        // Test whether function call can precede the definition.
        // Test whether the original order of the statements is maintained.
        // Test whether statements after the return statement are threaded.
        // These are unreachable, but must be threaded anyway.
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
a = foo()

def foo():
    return 5
    6
)")));
        tree->Accept(_thread_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);


        geoneric::AssignmentVertex const* assignment_vertex =
            dynamic_cast<geoneric::AssignmentVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(assignment_vertex);

        geoneric::FunctionCallVertex const* function_call_vertex =
            dynamic_cast<geoneric::FunctionCallVertex const*>(
                &(*assignment_vertex->expression()));
        BOOST_REQUIRE(function_call_vertex);

        geoneric::NameVertex const* name_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*assignment_vertex->target()));
        BOOST_REQUIRE(name_vertex);


        geoneric::FunctionDefinitionVertex const* function_definition_vertex =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[1]));
        BOOST_REQUIRE(function_definition_vertex);

        BOOST_REQUIRE_EQUAL(
            function_definition_vertex->scope()->statements().size(), 2u);
        geoneric::ReturnVertex const* return_vertex =
            dynamic_cast<geoneric::ReturnVertex const*>(
                &(*function_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);

        geoneric::NumberVertex<int64_t> const* number6_vertex =
            dynamic_cast<geoneric::NumberVertex<int64_t> const*>(
                &(*function_definition_vertex->scope()->statements()[1]));
        BOOST_REQUIRE(number6_vertex);

        BOOST_REQUIRE(return_vertex->expression());
        geoneric::NumberVertex<int64_t> const* number5_vertex =
            dynamic_cast<geoneric::NumberVertex<int64_t> const*>(
                &(*return_vertex->expression()));
        BOOST_REQUIRE(number5_vertex);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), function_call_vertex);
        BOOST_CHECK_EQUAL(function_call_vertex->successor(),
            function_definition_vertex);
        BOOST_CHECK_EQUAL(function_definition_vertex->successor(),
            function_definition_vertex->scope());
        BOOST_CHECK_EQUAL(function_definition_vertex->scope()->successor(),
            number5_vertex);
        BOOST_CHECK_EQUAL(number5_vertex->successor(), return_vertex);
        BOOST_CHECK_EQUAL(return_vertex->successor(),
            function_definition_vertex->scope()->sentinel());

        // number6_vertex is unreachable.
        BOOST_CHECK_EQUAL(number6_vertex->successor(),
            function_definition_vertex->scope()->sentinel());

        BOOST_REQUIRE(
            function_definition_vertex->scope()->sentinel()->has_successor());
        BOOST_CHECK_EQUAL(
            function_definition_vertex->scope()->sentinel()->successor(),
            name_vertex);
        BOOST_CHECK_EQUAL(name_vertex->successor(), assignment_vertex);
        BOOST_CHECK_EQUAL(assignment_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        // Add two arguments to the function and let the function return
        // the sum.
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
def sum(lhs, rhs):
    return lhs + rhs

s = sum(5, 6)
)")));
        tree->Accept(_thread_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);

        // def sum(lhs, rhs)
        geoneric::FunctionDefinitionVertex const* function_definition_vertex =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(function_definition_vertex);

        // (lhs, rhs)
        BOOST_REQUIRE_EQUAL(function_definition_vertex->arguments().size(), 2u);

        geoneric::NameVertex const* lhs_parameter_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*function_definition_vertex->arguments()[0]));
        BOOST_REQUIRE(lhs_parameter_vertex);

        geoneric::NameVertex const* rhs_parameter_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*function_definition_vertex->arguments()[1]));
        BOOST_REQUIRE(rhs_parameter_vertex);

        // return lhs + rhs
        BOOST_REQUIRE_EQUAL(
            function_definition_vertex->scope()->statements().size(), 1u);

        geoneric::ReturnVertex const* return_vertex =
            dynamic_cast<geoneric::ReturnVertex const*>(
                &(*function_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);
        BOOST_REQUIRE(return_vertex->expression());

        // lhs + rhs
        geoneric::OperatorVertex const* operator_vertex =
            dynamic_cast<geoneric::OperatorVertex const*>(
                &(*return_vertex->expression()));
        BOOST_REQUIRE(operator_vertex);
        BOOST_REQUIRE_EQUAL(operator_vertex->expressions().size(), 2u);

        // lhs
        geoneric::NameVertex const* lhs_argument_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*operator_vertex->expressions()[0]));
        BOOST_REQUIRE(lhs_argument_vertex);

        // rhs
        geoneric::NameVertex const* rhs_argument_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*operator_vertex->expressions()[1]));
        BOOST_REQUIRE(rhs_argument_vertex);

        // s = sum(5, 6)
        geoneric::AssignmentVertex const* assignment_vertex =
            dynamic_cast<geoneric::AssignmentVertex const*>(
                &(*tree->scope()->statements()[1]));
        BOOST_REQUIRE(assignment_vertex);

        geoneric::FunctionCallVertex const* function_call_vertex =
            dynamic_cast<geoneric::FunctionCallVertex const*>(
                &(*assignment_vertex->expression()));
        BOOST_REQUIRE(function_call_vertex);

        // (5, 6)
        BOOST_REQUIRE_EQUAL(function_call_vertex->expressions().size(), 2u);

        // 5
        geoneric::NumberVertex<int64_t> const* five_vertex =
            dynamic_cast<geoneric::NumberVertex<int64_t> const*>(
                &(*function_call_vertex->expressions()[0]));
        BOOST_REQUIRE(five_vertex);

        // 6
        geoneric::NumberVertex<int64_t> const* six_vertex =
            dynamic_cast<geoneric::NumberVertex<int64_t> const*>(
                &(*function_call_vertex->expressions()[1]));
        BOOST_REQUIRE(six_vertex);

        // s
        geoneric::NameVertex const* target_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*assignment_vertex->target()));
        BOOST_REQUIRE(target_vertex);

        // Threading.
        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), five_vertex);
        BOOST_CHECK_EQUAL(five_vertex->successor(), six_vertex);
        BOOST_CHECK_EQUAL(six_vertex->successor(), function_call_vertex);
        BOOST_CHECK_EQUAL(function_call_vertex->successor(),
            function_definition_vertex);
        BOOST_CHECK_EQUAL(function_definition_vertex->successor(),
            lhs_parameter_vertex);
        BOOST_CHECK_EQUAL(lhs_parameter_vertex->successor(),
            rhs_parameter_vertex);
        BOOST_CHECK_EQUAL(rhs_parameter_vertex->successor(),
            function_definition_vertex->scope());
        BOOST_CHECK_EQUAL(function_definition_vertex->scope()->successor(),
            lhs_argument_vertex);
        BOOST_CHECK_EQUAL(lhs_argument_vertex->successor(),
            rhs_argument_vertex);
        BOOST_CHECK_EQUAL(rhs_argument_vertex->successor(), operator_vertex);
        BOOST_CHECK_EQUAL(operator_vertex->successor(), return_vertex);
        BOOST_CHECK_EQUAL(return_vertex->successor(),
            function_definition_vertex->scope()->sentinel());
        BOOST_REQUIRE(
            function_definition_vertex->scope()->sentinel()->has_successor());
        BOOST_CHECK_EQUAL(
            function_definition_vertex->scope()->sentinel()->successor(),
            target_vertex);
        BOOST_CHECK_EQUAL(target_vertex->successor(), assignment_vertex);
        BOOST_CHECK_EQUAL(assignment_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        // Test nested function definitions.
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String(u8R"(
def foo(a, b):
    def bar(c, d):
        return c + d

    return bar(a, b)

result = foo(5, 6)
)")));
        tree->Accept(_thread_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);

        // def foo(a, b)
        geoneric::FunctionDefinitionVertex const* foo_definition_vertex =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(foo_definition_vertex);

        // (a, b)
        BOOST_REQUIRE_EQUAL(foo_definition_vertex->arguments().size(), 2u);

        geoneric::NameVertex const* a_parameter_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*foo_definition_vertex->arguments()[0]));
        BOOST_REQUIRE(a_parameter_vertex);

        geoneric::NameVertex const* b_parameter_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*foo_definition_vertex->arguments()[1]));
        BOOST_REQUIRE(b_parameter_vertex);

        BOOST_REQUIRE_EQUAL(
            foo_definition_vertex->scope()->statements().size(), 2u);

        // def bar(c, d):
        geoneric::FunctionDefinitionVertex const* bar_definition_vertex =
            dynamic_cast<geoneric::FunctionDefinitionVertex const*>(
                &(*foo_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(bar_definition_vertex);

        // (c, d)
        BOOST_REQUIRE_EQUAL(bar_definition_vertex->arguments().size(), 2u);

        geoneric::NameVertex const* c_parameter_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*bar_definition_vertex->arguments()[0]));
        BOOST_REQUIRE(c_parameter_vertex);

        geoneric::NameVertex const* d_parameter_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*bar_definition_vertex->arguments()[1]));
        BOOST_REQUIRE(d_parameter_vertex);

        BOOST_REQUIRE_EQUAL(
            bar_definition_vertex->scope()->statements().size(), 1u);

        geoneric::ReturnVertex const* bar_return_vertex =
            dynamic_cast<geoneric::ReturnVertex const*>(
                &(*bar_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(bar_return_vertex);
        BOOST_REQUIRE(bar_return_vertex->expression());

        // c + d
        geoneric::OperatorVertex const* operator_vertex =
            dynamic_cast<geoneric::OperatorVertex const*>(
                &(*bar_return_vertex->expression()));
        BOOST_REQUIRE(operator_vertex);
        BOOST_REQUIRE_EQUAL(operator_vertex->expressions().size(), 2u);

        // c
        geoneric::NameVertex const* c_argument_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*operator_vertex->expressions()[0]));
        BOOST_REQUIRE(c_argument_vertex);

        // d
        geoneric::NameVertex const* d_argument_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*operator_vertex->expressions()[1]));
        BOOST_REQUIRE(d_argument_vertex);

        // return bar(a, b)
        geoneric::ReturnVertex const* foo_return_vertex =
            dynamic_cast<geoneric::ReturnVertex const*>(
                &(*foo_definition_vertex->scope()->statements()[1]));
        BOOST_REQUIRE(foo_return_vertex);
        BOOST_REQUIRE(foo_return_vertex->expression());

        // bar(a, b)
        geoneric::FunctionCallVertex const* bar_call_vertex =
            dynamic_cast<geoneric::FunctionCallVertex const*>(
                &(*foo_return_vertex->expression()));
        BOOST_REQUIRE(bar_call_vertex);

        // (a, b)
        BOOST_REQUIRE_EQUAL(bar_call_vertex->expressions().size(), 2u);

        // a
        geoneric::NameVertex const* a_argument_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*bar_call_vertex->expressions()[0]));
        BOOST_REQUIRE(a_argument_vertex);

        // b
        geoneric::NameVertex const* b_argument_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*bar_call_vertex->expressions()[1]));
        BOOST_REQUIRE(b_argument_vertex);

        // result = foo(5, 6)
        geoneric::AssignmentVertex const* assignment_vertex =
            dynamic_cast<geoneric::AssignmentVertex const*>(
                &(*tree->scope()->statements()[1]));
        BOOST_REQUIRE(assignment_vertex);

        geoneric::FunctionCallVertex const* foo_call_vertex =
            dynamic_cast<geoneric::FunctionCallVertex const*>(
                &(*assignment_vertex->expression()));
        BOOST_REQUIRE(foo_call_vertex);

        // (5, 6)
        BOOST_REQUIRE_EQUAL(foo_call_vertex->expressions().size(), 2u);

        // 5
        geoneric::NumberVertex<int64_t> const* five_vertex =
            dynamic_cast<geoneric::NumberVertex<int64_t> const*>(
                &(*foo_call_vertex->expressions()[0]));
        BOOST_REQUIRE(five_vertex);

        // 6
        geoneric::NumberVertex<int64_t> const* six_vertex =
            dynamic_cast<geoneric::NumberVertex<int64_t> const*>(
                &(*foo_call_vertex->expressions()[1]));
        BOOST_REQUIRE(six_vertex);

        // result
        geoneric::NameVertex const* target_vertex =
            dynamic_cast<geoneric::NameVertex const*>(
                &(*assignment_vertex->target()));
        BOOST_REQUIRE(target_vertex);

        // Threading.
        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), five_vertex);
        BOOST_CHECK_EQUAL(five_vertex->successor(), six_vertex);
        BOOST_CHECK_EQUAL(six_vertex->successor(), foo_call_vertex);

        BOOST_CHECK_EQUAL(foo_call_vertex->successor(), foo_definition_vertex);
        BOOST_CHECK_EQUAL(foo_definition_vertex->successor(),
            a_parameter_vertex);
        BOOST_CHECK_EQUAL(a_parameter_vertex->successor(), b_parameter_vertex);
        BOOST_CHECK_EQUAL(b_parameter_vertex->successor(),
            foo_definition_vertex->scope());
        BOOST_CHECK_EQUAL(foo_definition_vertex->scope()->successor(),
            a_argument_vertex);
        BOOST_CHECK_EQUAL(a_argument_vertex->successor(),
            b_argument_vertex);
        BOOST_CHECK_EQUAL(b_argument_vertex->successor(), bar_call_vertex);

        BOOST_CHECK_EQUAL(bar_call_vertex->successor(), bar_definition_vertex);
        BOOST_CHECK_EQUAL(bar_definition_vertex->successor(),
            c_parameter_vertex);
        BOOST_CHECK_EQUAL(c_parameter_vertex->successor(), d_parameter_vertex);
        BOOST_CHECK_EQUAL(d_parameter_vertex->successor(),
            bar_definition_vertex->scope());
        BOOST_CHECK_EQUAL(bar_definition_vertex->scope()->successor(),
            c_argument_vertex);
        BOOST_CHECK_EQUAL(c_argument_vertex->successor(),
            d_argument_vertex);
        BOOST_CHECK_EQUAL(d_argument_vertex->successor(), operator_vertex);

        BOOST_CHECK_EQUAL(operator_vertex->successor(), bar_return_vertex);
        BOOST_CHECK_EQUAL(bar_return_vertex->successor(),
            bar_definition_vertex->scope()->sentinel());
        BOOST_REQUIRE(
            bar_definition_vertex->scope()->sentinel()->has_successor());
        BOOST_CHECK_EQUAL(
            bar_definition_vertex->scope()->sentinel()->successor(),
            foo_return_vertex);
        BOOST_CHECK_EQUAL(foo_return_vertex->successor(),
            foo_definition_vertex->scope()->sentinel());
        BOOST_REQUIRE(
            foo_definition_vertex->scope()->sentinel()->has_successor());

        BOOST_CHECK_EQUAL(
            foo_definition_vertex->scope()->sentinel()->successor(),
            target_vertex);
        BOOST_CHECK_EQUAL(target_vertex->successor(), assignment_vertex);
        BOOST_CHECK_EQUAL(assignment_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }
}


BOOST_AUTO_TEST_CASE(visit_subscript)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("a[b]")));
        tree->Accept(_thread_visitor);

        geoneric::SubscriptVertex const* subscript =
            dynamic_cast<geoneric::SubscriptVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(subscript);

        geoneric::AstVertex const* a = &(*subscript->expression());
        geoneric::AstVertex const* b = &(*subscript->selection());

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), a);
        BOOST_CHECK_EQUAL(a->successor(), b);
        BOOST_CHECK_EQUAL(b->successor(), subscript);
        BOOST_CHECK_EQUAL(subscript->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }
}


BOOST_AUTO_TEST_CASE(visit_attribute)
{
    std::shared_ptr<geoneric::ModuleVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            geoneric::String("a.b")));
        tree->Accept(_thread_visitor);

        geoneric::AttributeVertex const* attribute =
            dynamic_cast<geoneric::AttributeVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(attribute);

        geoneric::AstVertex const* a = &(*attribute->expression());

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), a);
        BOOST_CHECK_EQUAL(a->successor(), attribute);
        BOOST_CHECK_EQUAL(attribute->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }
}

BOOST_AUTO_TEST_SUITE_END()
