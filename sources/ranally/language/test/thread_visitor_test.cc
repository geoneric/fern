#define BOOST_TEST_MODULE ranally language
#include <boost/test/included/unit_test.hpp>
#include "ranally/core/string.h"
#include "ranally/language/algebra_parser.h"
#include "ranally/language/assignment_vertex.h"
#include "ranally/language/function_vertex.h"
#include "ranally/language/if_vertex.h"
#include "ranally/language/operator_vertex.h"
#include "ranally/language/script_vertex.h"
#include "ranally/language/thread_visitor.h"
#include "ranally/language/xml_parser.h"


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

    tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String("")));
    tree->Accept(_visitor);
    BOOST_CHECK_EQUAL(tree->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
        "a")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* vertex_a = &(*tree->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse(_algebra_parser.parse_string(
        ranally::String("a = b")));
    tree->Accept(_visitor);

    ranally::AssignmentVertex const* assignment =
        dynamic_cast<ranally::AssignmentVertex const*>(
            &(*tree->statements()[0]));
    ranally::SyntaxVertex const* vertex_a = &(*assignment->target());
    ranally::SyntaxVertex const* vertex_b =
        &(*assignment->expression());

    BOOST_CHECK_EQUAL(tree->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), assignment);
    BOOST_CHECK_EQUAL(assignment->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
        "\"five\"")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* string_vertex =
        &(*tree->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), string_vertex);
    BOOST_CHECK_EQUAL(string_vertex->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
        "5")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* number_vertex = &(*tree->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), number_vertex);
    BOOST_CHECK_EQUAL(number_vertex->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(
            ranally::String("f()")));
        tree->Accept(_visitor);

        ranally::SyntaxVertex const* function_vertex =
            &(*tree->statements()[0]);

        BOOST_CHECK_EQUAL(tree->successor(), function_vertex);
        BOOST_CHECK_EQUAL(function_vertex->successor(), &(*tree));
    }

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
            "f(1, \"2\", three, four())")));
        tree->Accept(_visitor);

        ranally::FunctionVertex const* function_vertex =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*function_vertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
            &(*function_vertex->expressions()[1]);
        ranally::SyntaxVertex const* vertex3 =
            &(*function_vertex->expressions()[2]);
        ranally::SyntaxVertex const* vertex4 =
            &(*function_vertex->expressions()[3]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), vertex3);
        BOOST_CHECK_EQUAL(vertex3->successor(), vertex4);
        BOOST_CHECK_EQUAL(vertex4->successor(), function_vertex);
        BOOST_CHECK_EQUAL(function_vertex->successor(), &(*tree));
    }
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(
            ranally::String("-a")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operator_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operator_vertex->expressions()[0]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), operator_vertex);
        BOOST_CHECK_EQUAL(operator_vertex->successor(), &(*tree));
    }

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(
            ranally::String("a + b")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operator_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operator_vertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
            &(*operator_vertex->expressions()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), operator_vertex);
        BOOST_CHECK_EQUAL(operator_vertex->successor(), &(*tree));
    }

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
            "-(a + b)")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operator1_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->statements()[0]));
        ranally::OperatorVertex const* operator2_vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*operator1_vertex->expressions()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operator2_vertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
            &(*operator2_vertex->expressions()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), operator2_vertex);
        BOOST_CHECK_EQUAL(operator2_vertex->successor(), operator1_vertex);
        BOOST_CHECK_EQUAL(operator1_vertex->successor(), &(*tree));
    }
}


BOOST_AUTO_TEST_CASE(visit_multiple_statement)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
        "a;b;c")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* vertex_a = &(*tree->statements()[0]);
    ranally::SyntaxVertex const* vertex_b = &(*tree->statements()[1]);
    ranally::SyntaxVertex const* vertex_c = &(*tree->statements()[2]);

    BOOST_CHECK_EQUAL(tree->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
    BOOST_CHECK_EQUAL(vertex_c->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_nested_expressions)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
        "a = b + c")));
    tree->Accept(_visitor);

    ranally::AssignmentVertex const* assignment =
        dynamic_cast<ranally::AssignmentVertex const*>(
            &(*tree->statements()[0]));
    ranally::SyntaxVertex const* vertex_a = &(*assignment->target());
    ranally::OperatorVertex const* addition =
        dynamic_cast<ranally::OperatorVertex const*>(
            &(*assignment->expression()));
    ranally::SyntaxVertex const* vertex_b =
        &(*addition->expressions()[0]);
    ranally::SyntaxVertex const* vertex_c =
        &(*addition->expressions()[1]);

    BOOST_CHECK_EQUAL(tree->successor(), vertex_b);
    BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
    BOOST_CHECK_EQUAL(vertex_c->successor(), addition);
    BOOST_CHECK_EQUAL(addition->successor(), vertex_a);
    BOOST_CHECK_EQUAL(vertex_a->successor(), assignment);
    BOOST_CHECK_EQUAL(assignment->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
            "if a:\n"
            "    b\n"
            "    c")));
        tree->Accept(_visitor);

        ranally::IfVertex const* if_vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex_a =
            &(*if_vertex->condition());
        ranally::SyntaxVertex const* vertex_b =
            &(*if_vertex->true_statements()[0]);
        ranally::SyntaxVertex const* vertex_c =
            &(*if_vertex->true_statements()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex_a);
        BOOST_CHECK_EQUAL(vertex_a->successor(), if_vertex);
        BOOST_CHECK_EQUAL(if_vertex->successor(0), vertex_b);
        BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
        BOOST_CHECK_EQUAL(vertex_c->successor(), if_vertex);
        BOOST_CHECK_EQUAL(if_vertex->successor(1), &(*tree));
    }

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
            "if a:\n"
            "    b\n"
            "    c\n"
            "elif d:\n"
            "    e\n"
            "    f\n")));
        tree->Accept(_visitor);

        // True block first if.
        ranally::IfVertex const* if1_vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex_a =
            &(*if1_vertex->condition());
        ranally::SyntaxVertex const* vertex_b =
            &(*if1_vertex->true_statements()[0]);
        ranally::SyntaxVertex const* vertex_c =
            &(*if1_vertex->true_statements()[1]);

        // True block second if.
        ranally::IfVertex const* if2_vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*if1_vertex->false_statements()[0]));
        ranally::SyntaxVertex const* vertex_d =
            &(*if2_vertex->condition());
        ranally::SyntaxVertex const* vertex_e =
            &(*if2_vertex->true_statements()[0]);
        ranally::SyntaxVertex const* vertex_f =
            &(*if2_vertex->true_statements()[1]);

        // True block first if.
        BOOST_CHECK_EQUAL(tree->successor(), vertex_a);
        BOOST_CHECK_EQUAL(vertex_a->successor(), if1_vertex);
        BOOST_CHECK_EQUAL(if1_vertex->successor(0), vertex_b);
        BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
        BOOST_CHECK_EQUAL(vertex_c->successor(), if1_vertex);

        // True block second if.
        BOOST_CHECK_EQUAL(if1_vertex->successor(1), vertex_d);
        BOOST_CHECK_EQUAL(vertex_d->successor(), if2_vertex);
        BOOST_CHECK_EQUAL(if2_vertex->successor(0), vertex_e);
        BOOST_CHECK_EQUAL(vertex_e->successor(), vertex_f);
        BOOST_CHECK_EQUAL(vertex_f->successor(), if2_vertex);

        BOOST_CHECK_EQUAL(if2_vertex->successor(1), if1_vertex);

        BOOST_CHECK_EQUAL(if1_vertex->successor(2), &(*tree));
    }

    {
        tree = _xml_parser.parse(_algebra_parser.parse_string(ranally::String(
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
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex_a =
            &(*if1_vertex->condition());
        ranally::SyntaxVertex const* vertex_b =
            &(*if1_vertex->true_statements()[0]);
        ranally::SyntaxVertex const* vertex_c =
            &(*if1_vertex->true_statements()[1]);

        // True block second if.
        ranally::IfVertex const* if2_vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*if1_vertex->false_statements()[0]));
        ranally::SyntaxVertex const* vertex_d =
            &(*if2_vertex->condition());
        ranally::SyntaxVertex const* vertex_e =
            &(*if2_vertex->true_statements()[0]);
        ranally::SyntaxVertex const* vertex_f =
            &(*if2_vertex->true_statements()[1]);

        // False block second if.
        ranally::SyntaxVertex const* vertex_g =
            &(*if2_vertex->false_statements()[0]);
        ranally::SyntaxVertex const* vertex_h =
            &(*if2_vertex->false_statements()[1]);

        // True block first if.
        BOOST_CHECK_EQUAL(tree->successor(), vertex_a);
        BOOST_CHECK_EQUAL(vertex_a->successor(), if1_vertex);
        BOOST_CHECK_EQUAL(if1_vertex->successor(0), vertex_b);
        BOOST_CHECK_EQUAL(vertex_b->successor(), vertex_c);
        BOOST_CHECK_EQUAL(vertex_c->successor(), if1_vertex);

        // True block second if.
        BOOST_CHECK_EQUAL(if1_vertex->successor(1), vertex_d);
        BOOST_CHECK_EQUAL(vertex_d->successor(), if2_vertex);
        BOOST_CHECK_EQUAL(if2_vertex->successor(0), vertex_e);
        BOOST_CHECK_EQUAL(vertex_e->successor(), vertex_f);
        BOOST_CHECK_EQUAL(vertex_f->successor(), if2_vertex);

        // False block second if.
        BOOST_CHECK_EQUAL(if2_vertex->successor(1), vertex_g);
        BOOST_CHECK_EQUAL(vertex_g->successor(), vertex_h);
        BOOST_CHECK_EQUAL(vertex_h->successor(), if2_vertex);

        BOOST_CHECK_EQUAL(if2_vertex->successor(2), if1_vertex);

        BOOST_CHECK_EQUAL(if1_vertex->successor(2), &(*tree));
    }
}


BOOST_AUTO_TEST_CASE(visit_while)
{
}

BOOST_AUTO_TEST_SUITE_END()
