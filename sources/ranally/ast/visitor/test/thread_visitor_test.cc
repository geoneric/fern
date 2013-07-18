#define BOOST_TEST_MODULE ranally ast
#include <typeinfo>
//    std::cout << typeid(*tree->scope()->successor()).name() << std::endl;
#include <boost/test/unit_test.hpp>
#include "ranally/core/string.h"
#include "ranally/script/algebra_parser.h"
#include "ranally/ast/core/vertices.h"
#include "ranally/ast/visitor/thread_visitor.h"
#include "ranally/ast/xml/xml_parser.h"


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
          _visitor()
    {
    }

protected:

    ranally::AlgebraParser _algebra_parser;

    ranally::XmlParser _xml_parser;

    ranally::ThreadVisitor _visitor;

};


BOOST_FIXTURE_TEST_SUITE(thread_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("")));
    tree->Accept(_visitor);
    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("a")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* vertex_a = &(*tree->scope()->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("a = b")));
    tree->Accept(_visitor);

    ranally::AssignmentVertex const* assignment =
        dynamic_cast<ranally::AssignmentVertex const*>(
            &(*tree->scope()->statements()[0]));
    ranally::SyntaxVertex const* vertex_a = &(*assignment->target());
    ranally::SyntaxVertex const* vertex_b = &(*assignment->expression());

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), assignment);
    BOOST_CHECK_EQUAL(assignment->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("\"five\"")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* string_vertex =
        &(*tree->scope()->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), string_vertex);
    BOOST_CHECK_EQUAL(string_vertex->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("5")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* number_vertex =
        &(*tree->scope()->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), number_vertex);
    BOOST_CHECK_EQUAL(number_vertex->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("f()")));
        tree->Accept(_visitor);

        ranally::SyntaxVertex const* function_vertex =
            &(*tree->scope()->statements()[0]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), function_vertex);
        BOOST_CHECK_EQUAL(function_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("f(1, \"2\", three, four())")));
        tree->Accept(_visitor);

        ranally::FunctionVertex const* function_vertex =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*tree->scope()->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*function_vertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
            &(*function_vertex->expressions()[1]);
        ranally::SyntaxVertex const* vertex3 =
            &(*function_vertex->expressions()[2]);
        ranally::SyntaxVertex const* vertex4 =
            &(*function_vertex->expressions()[3]);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), vertex3);
        BOOST_CHECK_EQUAL(vertex3->successor(), vertex4);
        BOOST_CHECK_EQUAL(vertex4->successor(), function_vertex);
        BOOST_CHECK_EQUAL(function_vertex->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
    }
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String("-a")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operator_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->scope()->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
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
            ranally::String("a + b")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operator_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->scope()->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operator_vertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
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
            ranally::String("-(a + b)")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operator1_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->scope()->statements()[0]));
        ranally::OperatorVertex const* operator2_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*operator1_vertex->expressions()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operator2_vertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
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
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("a;b;c")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* vertex_a = &(*tree->scope()->statements()[0]);
    ranally::SyntaxVertex const* vertex_b = &(*tree->scope()->statements()[1]);
    ranally::SyntaxVertex const* vertex_c = &(*tree->scope()->statements()[2]);

    BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
    BOOST_CHECK_EQUAL(tree->scope()->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
    BOOST_CHECK_EQUAL(vertex_c->successor(), tree->scope()->sentinel());
    BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);
}


BOOST_AUTO_TEST_CASE(visit_nested_expressions)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse_string(_algebra_parser.parse_string(
        ranally::String("a = b + c")));
    tree->Accept(_visitor);

    ranally::AssignmentVertex const* assignment =
        dynamic_cast<ranally::AssignmentVertex const*>(
            &(*tree->scope()->statements()[0]));
    ranally::SyntaxVertex const* vertex_a = &(*assignment->target());
    ranally::OperatorVertex const* addition =
        dynamic_cast<ranally::OperatorVertex const*>(
            &(*assignment->expression()));
    ranally::SyntaxVertex const* vertex_b =
        &(*addition->expressions()[0]);
    ranally::SyntaxVertex const* vertex_c =
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
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(
                "if a:\n"
                "    b\n"
                "    c")));
        tree->Accept(_visitor);

        ranally::IfVertex const* if_vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->scope()->statements()[0]));
        ranally::SyntaxVertex const* vertex_a =
            &(*if_vertex->condition());
        ranally::SyntaxVertex const* vertex_b =
            &(*if_vertex->true_scope()->statements()[0]);
        ranally::SyntaxVertex const* vertex_c =
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
            ranally::String(
                "if a:\n"
                "    b\n"
                "    c\n"
                "elif d:\n"
                "    e\n"
                "    f\n"
                "else:\n"
                "    g\n"
                "    h\n")));
        tree->Accept(_visitor);

        // True block first if.
        ranally::IfVertex const* if1_vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->scope()->statements()[0]));
        ranally::SyntaxVertex const* vertex_a =
            &(*if1_vertex->condition());
        ranally::SyntaxVertex const* vertex_b =
            &(*if1_vertex->true_scope()->statements()[0]);
        ranally::SyntaxVertex const* vertex_c =
            &(*if1_vertex->true_scope()->statements()[1]);

        // True block second if.
        ranally::IfVertex const* if2_vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*if1_vertex->false_scope()->statements()[0]));
        ranally::SyntaxVertex const* vertex_d =
            &(*if2_vertex->condition());
        ranally::SyntaxVertex const* vertex_e =
            &(*if2_vertex->true_scope()->statements()[0]);
        ranally::SyntaxVertex const* vertex_f =
            &(*if2_vertex->true_scope()->statements()[1]);

        // False block second if.
        ranally::SyntaxVertex const* vertex_g =
            &(*if2_vertex->false_scope()->statements()[0]);
        ranally::SyntaxVertex const* vertex_h =
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
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(u8R"(
def foo():
    return
)")));
        tree->Accept(_visitor);

        // The defined function isn't called, so the script is in effect
        // empty. This doesn't mean the definition isn't in the script. It
        // means threading hasn't connected to the function.
        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        ranally::FunctionDefinitionVertex const* function_definition_vertex =
            dynamic_cast<ranally::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(function_definition_vertex);

        BOOST_REQUIRE_EQUAL(
            function_definition_vertex->scope()->statements().size(), 1u);
        ranally::ReturnVertex const* return_vertex =
            dynamic_cast<ranally::ReturnVertex const*>(
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
            ranally::String(u8R"(
def foo():
    return 5
)")));
        tree->Accept(_visitor);

        // The defined function isn't called, so the script is in effect
        // empty. This doesn't mean the definition isn't in the script. It
        // means threading hasn't connected to the function.
        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(),
            tree->scope()->sentinel());
        BOOST_CHECK_EQUAL(tree->scope()->sentinel()->successor(), tree);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 1u);
        ranally::FunctionDefinitionVertex const* vertex_foo =
            dynamic_cast<ranally::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(vertex_foo);

        BOOST_REQUIRE_EQUAL(vertex_foo->scope()->statements().size(), 1u);
        ranally::ReturnVertex const* return_vertex =
            dynamic_cast<ranally::ReturnVertex const*>(
                &(*vertex_foo->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);

        BOOST_REQUIRE(return_vertex->expression());
        ranally::SyntaxVertex const* number_vertex =
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
            ranally::String(u8R"(
def foo():
    return 5

a = foo())")));
        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);

        ranally::FunctionDefinitionVertex const* vertex_foo =
            dynamic_cast<ranally::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(vertex_foo);

        BOOST_REQUIRE_EQUAL(vertex_foo->scope()->statements().size(), 1u);
        ranally::ReturnVertex const* return_vertex =
            dynamic_cast<ranally::ReturnVertex const*>(
                &(*vertex_foo->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);

        BOOST_REQUIRE(return_vertex->expression());
        ranally::NumberVertex<int64_t> const* number_vertex =
            dynamic_cast<ranally::NumberVertex<int64_t> const*>(
                &(*return_vertex->expression()));

        ranally::AssignmentVertex const* assignment_vertex =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->scope()->statements()[1]));
        BOOST_REQUIRE(assignment_vertex);

        ranally::FunctionVertex const* function_vertex =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment_vertex->expression()));
        BOOST_REQUIRE(function_vertex);

        ranally::NameVertex const* name_vertex =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_vertex->target()));
        BOOST_REQUIRE(name_vertex);

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), function_vertex);
        BOOST_CHECK_EQUAL(function_vertex->successor(), vertex_foo);
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
            ranally::String(u8R"(
def foo():
    bar

foo())")));
        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);

        ranally::FunctionDefinitionVertex const* function_definition_vertex =
            dynamic_cast<ranally::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(function_definition_vertex);

        BOOST_REQUIRE_EQUAL(
            function_definition_vertex->scope()->statements().size(), 1u);
        ranally::NameVertex const* name_vertex =
            dynamic_cast<ranally::NameVertex const*>(
                &(*function_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(name_vertex);

        ranally::FunctionVertex const* function_call_vertex =
            dynamic_cast<ranally::FunctionVertex const*>(
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
        tree = _xml_parser.parse_string(_algebra_parser.parse_string(
            ranally::String(u8R"(
a = foo()

def foo():
    return 5
)")));
        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(tree->scope()->statements().size(), 2u);


        ranally::AssignmentVertex const* assignment_vertex =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->scope()->statements()[0]));
        BOOST_REQUIRE(assignment_vertex);

        ranally::FunctionVertex const* function_call_vertex =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment_vertex->expression()));
        BOOST_REQUIRE(function_call_vertex);

        ranally::NameVertex const* name_vertex =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment_vertex->target()));
        BOOST_REQUIRE(name_vertex);


        ranally::FunctionDefinitionVertex const* function_definition_vertex =
            dynamic_cast<ranally::FunctionDefinitionVertex const*>(
                &(*tree->scope()->statements()[1]));
        BOOST_REQUIRE(function_definition_vertex);

        BOOST_REQUIRE_EQUAL(
            function_definition_vertex->scope()->statements().size(), 1u);
        ranally::ReturnVertex const* return_vertex =
            dynamic_cast<ranally::ReturnVertex const*>(
                &(*function_definition_vertex->scope()->statements()[0]));
        BOOST_REQUIRE(return_vertex);

        BOOST_REQUIRE(return_vertex->expression());
        ranally::NumberVertex<int64_t> const* number_vertex =
            dynamic_cast<ranally::NumberVertex<int64_t> const*>(
                &(*return_vertex->expression()));

        BOOST_CHECK_EQUAL(tree->successor(), tree->scope());
        BOOST_CHECK_EQUAL(tree->scope()->successor(), function_call_vertex);
        BOOST_CHECK_EQUAL(function_call_vertex->successor(),
            function_definition_vertex);
        BOOST_CHECK_EQUAL(function_definition_vertex->successor(),
            function_definition_vertex->scope());
        BOOST_CHECK_EQUAL(function_definition_vertex->scope()->successor(),
            number_vertex);
        BOOST_CHECK_EQUAL(number_vertex->successor(), return_vertex);
        BOOST_CHECK_EQUAL(return_vertex->successor(),
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

    // TODO Test function with return statement and subsequent statements.
    // TODO Add two arguments to the function and let the function return
    //      the sum.
    // TODO Rename SyntaxVertex to AstVertex. The vertices are more related
    //             to the AST than to the syntax of the language. Some of
    //             the vertices have no equivalent in the language
    //             (SentinelVertex).
    // TODO Rename ScriptVertex to ModuleVertex.
    // TODO Rename FunctionVertex to CallVertex?
    //      Rename FunctionDefinitionVertex to FunctionVertex?
    // TODO Add sentinel to script, if, while statements.
    // TODO Test nested function definition.

}

BOOST_AUTO_TEST_SUITE_END()
