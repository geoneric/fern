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
        : _algebraParser(),
          _xmlParser(),
          _visitor()
    {
    }

protected:

    ranally::AlgebraParser _algebraParser;

    ranally::XmlParser _xmlParser;

    ranally::ThreadVisitor _visitor;

};


BOOST_FIXTURE_TEST_SUITE(thread_visitor, Support)


BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String("")));
    tree->Accept(_visitor);
    BOOST_CHECK_EQUAL(tree->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String("a")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* vertexA = &(*tree->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), vertexA);
    BOOST_CHECK_EQUAL(vertexA->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(
        ranally::String("a = b")));
    tree->Accept(_visitor);

    ranally::AssignmentVertex const* assignment =
        dynamic_cast<ranally::AssignmentVertex const*>(
            &(*tree->statements()[0]));
    ranally::SyntaxVertex const* vertexA = &(*assignment->target());
    ranally::SyntaxVertex const* vertexB =
        &(*assignment->expression());

    BOOST_CHECK_EQUAL(tree->successor(), vertexB);
    BOOST_CHECK_EQUAL(vertexB->successor(), vertexA);
    BOOST_CHECK_EQUAL(vertexA->successor(), assignment);
    BOOST_CHECK_EQUAL(assignment->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_string)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
        "\"five\"")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* stringVertex =
        &(*tree->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), stringVertex);
    BOOST_CHECK_EQUAL(stringVertex->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_number)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String("5")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* numberVertex =
        &(*tree->statements()[0]);

    BOOST_CHECK_EQUAL(tree->successor(), numberVertex);
    BOOST_CHECK_EQUAL(numberVertex->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_function)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(
            ranally::String("f()")));
        tree->Accept(_visitor);

        ranally::SyntaxVertex const* functionVertex =
            &(*tree->statements()[0]);

        BOOST_CHECK_EQUAL(tree->successor(), functionVertex);
        BOOST_CHECK_EQUAL(functionVertex->successor(), &(*tree));
    }

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "f(1, \"2\", three, four())")));
        tree->Accept(_visitor);

        ranally::FunctionVertex const* functionVertex =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*functionVertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
            &(*functionVertex->expressions()[1]);
        ranally::SyntaxVertex const* vertex3 =
            &(*functionVertex->expressions()[2]);
        ranally::SyntaxVertex const* vertex4 =
            &(*functionVertex->expressions()[3]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), vertex3);
        BOOST_CHECK_EQUAL(vertex3->successor(), vertex4);
        BOOST_CHECK_EQUAL(vertex4->successor(), functionVertex);
        BOOST_CHECK_EQUAL(functionVertex->successor(), &(*tree));
    }
}


BOOST_AUTO_TEST_CASE(visit_operator)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(
            ranally::String("-a")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operatorVertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operatorVertex->expressions()[0]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), operatorVertex);
        BOOST_CHECK_EQUAL(operatorVertex->successor(), &(*tree));
    }

    {
        tree = _xmlParser.parse(_algebraParser.parseString(
            ranally::String("a + b")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operatorVertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operatorVertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
            &(*operatorVertex->expressions()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), operatorVertex);
        BOOST_CHECK_EQUAL(operatorVertex->successor(), &(*tree));
    }

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "-(a + b)")));
        tree->Accept(_visitor);

        ranally::OperatorVertex const* operator1Vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*tree->statements()[0]));
        ranally::OperatorVertex const* operator2Vertex =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*operator1Vertex->expressions()[0]));
        ranally::SyntaxVertex const* vertex1 =
            &(*operator2Vertex->expressions()[0]);
        ranally::SyntaxVertex const* vertex2 =
            &(*operator2Vertex->expressions()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), vertex1);
        BOOST_CHECK_EQUAL(vertex1->successor(), vertex2);
        BOOST_CHECK_EQUAL(vertex2->successor(), operator2Vertex);
        BOOST_CHECK_EQUAL(operator2Vertex->successor(), operator1Vertex);
        BOOST_CHECK_EQUAL(operator1Vertex->successor(), &(*tree));
    }
}


BOOST_AUTO_TEST_CASE(visit_multiple_statement)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
        "a;b;c")));
    tree->Accept(_visitor);

    ranally::SyntaxVertex const* vertexA = &(*tree->statements()[0]);
    ranally::SyntaxVertex const* vertexB = &(*tree->statements()[1]);
    ranally::SyntaxVertex const* vertexC = &(*tree->statements()[2]);

    BOOST_CHECK_EQUAL(tree->successor(), vertexA);
    BOOST_CHECK_EQUAL(vertexA->successor(), vertexB);
    BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
    BOOST_CHECK_EQUAL(vertexC->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_nested_expressions)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
        "a = b + c")));
    tree->Accept(_visitor);

    ranally::AssignmentVertex const* assignment =
        dynamic_cast<ranally::AssignmentVertex const*>(
            &(*tree->statements()[0]));
    ranally::SyntaxVertex const* vertexA = &(*assignment->target());
    ranally::OperatorVertex const* addition =
        dynamic_cast<ranally::OperatorVertex const*>(
            &(*assignment->expression()));
    ranally::SyntaxVertex const* vertexB =
        &(*addition->expressions()[0]);
    ranally::SyntaxVertex const* vertexC =
        &(*addition->expressions()[1]);

    BOOST_CHECK_EQUAL(tree->successor(), vertexB);
    BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
    BOOST_CHECK_EQUAL(vertexC->successor(), addition);
    BOOST_CHECK_EQUAL(addition->successor(), vertexA);
    BOOST_CHECK_EQUAL(vertexA->successor(), assignment);
    BOOST_CHECK_EQUAL(assignment->successor(), &(*tree));
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "if a:\n"
            "    b\n"
            "    c")));
        tree->Accept(_visitor);

        ranally::IfVertex const* ifVertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertexA =
            &(*ifVertex->condition());
        ranally::SyntaxVertex const* vertexB =
            &(*ifVertex->trueStatements()[0]);
        ranally::SyntaxVertex const* vertexC =
            &(*ifVertex->trueStatements()[1]);

        BOOST_CHECK_EQUAL(tree->successor(), vertexA);
        BOOST_CHECK_EQUAL(vertexA->successor(), ifVertex);
        BOOST_CHECK_EQUAL(ifVertex->successor(0), vertexB);
        BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
        BOOST_CHECK_EQUAL(vertexC->successor(), ifVertex);
        BOOST_CHECK_EQUAL(ifVertex->successor(1), &(*tree));
    }

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "if a:\n"
            "    b\n"
            "    c\n"
            "elif d:\n"
            "    e\n"
            "    f\n")));
        tree->Accept(_visitor);

        // True block first if.
        ranally::IfVertex const* if1Vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertexA =
            &(*if1Vertex->condition());
        ranally::SyntaxVertex const* vertexB =
            &(*if1Vertex->trueStatements()[0]);
        ranally::SyntaxVertex const* vertexC =
            &(*if1Vertex->trueStatements()[1]);

        // True block second if.
        ranally::IfVertex const* if2Vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*if1Vertex->falseStatements()[0]));
        ranally::SyntaxVertex const* vertexD =
            &(*if2Vertex->condition());
        ranally::SyntaxVertex const* vertexE =
            &(*if2Vertex->trueStatements()[0]);
        ranally::SyntaxVertex const* vertexF =
            &(*if2Vertex->trueStatements()[1]);

        // True block first if.
        BOOST_CHECK_EQUAL(tree->successor(), vertexA);
        BOOST_CHECK_EQUAL(vertexA->successor(), if1Vertex);
        BOOST_CHECK_EQUAL(if1Vertex->successor(0), vertexB);
        BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
        BOOST_CHECK_EQUAL(vertexC->successor(), if1Vertex);

        // True block second if.
        BOOST_CHECK_EQUAL(if1Vertex->successor(1), vertexD);
        BOOST_CHECK_EQUAL(vertexD->successor(), if2Vertex);
        BOOST_CHECK_EQUAL(if2Vertex->successor(0), vertexE);
        BOOST_CHECK_EQUAL(vertexE->successor(), vertexF);
        BOOST_CHECK_EQUAL(vertexF->successor(), if2Vertex);

        BOOST_CHECK_EQUAL(if2Vertex->successor(1), if1Vertex);

        BOOST_CHECK_EQUAL(if1Vertex->successor(2), &(*tree));
    }

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
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
        ranally::IfVertex const* if1Vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->statements()[0]));
        ranally::SyntaxVertex const* vertexA =
            &(*if1Vertex->condition());
        ranally::SyntaxVertex const* vertexB =
            &(*if1Vertex->trueStatements()[0]);
        ranally::SyntaxVertex const* vertexC =
            &(*if1Vertex->trueStatements()[1]);

        // True block second if.
        ranally::IfVertex const* if2Vertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*if1Vertex->falseStatements()[0]));
        ranally::SyntaxVertex const* vertexD =
            &(*if2Vertex->condition());
        ranally::SyntaxVertex const* vertexE =
            &(*if2Vertex->trueStatements()[0]);
        ranally::SyntaxVertex const* vertexF =
            &(*if2Vertex->trueStatements()[1]);

        // False block second if.
        ranally::SyntaxVertex const* vertexG =
            &(*if2Vertex->falseStatements()[0]);
        ranally::SyntaxVertex const* vertexH =
            &(*if2Vertex->falseStatements()[1]);

        // True block first if.
        BOOST_CHECK_EQUAL(tree->successor(), vertexA);
        BOOST_CHECK_EQUAL(vertexA->successor(), if1Vertex);
        BOOST_CHECK_EQUAL(if1Vertex->successor(0), vertexB);
        BOOST_CHECK_EQUAL(vertexB->successor(), vertexC);
        BOOST_CHECK_EQUAL(vertexC->successor(), if1Vertex);

        // True block second if.
        BOOST_CHECK_EQUAL(if1Vertex->successor(1), vertexD);
        BOOST_CHECK_EQUAL(vertexD->successor(), if2Vertex);
        BOOST_CHECK_EQUAL(if2Vertex->successor(0), vertexE);
        BOOST_CHECK_EQUAL(vertexE->successor(), vertexF);
        BOOST_CHECK_EQUAL(vertexF->successor(), if2Vertex);

        // False block second if.
        BOOST_CHECK_EQUAL(if2Vertex->successor(1), vertexG);
        BOOST_CHECK_EQUAL(vertexG->successor(), vertexH);
        BOOST_CHECK_EQUAL(vertexH->successor(), if2Vertex);

        BOOST_CHECK_EQUAL(if2Vertex->successor(2), if1Vertex);

        BOOST_CHECK_EQUAL(if1Vertex->successor(2), &(*tree));
    }
}


BOOST_AUTO_TEST_CASE(visit_while)
{
}

BOOST_AUTO_TEST_SUITE_END()
