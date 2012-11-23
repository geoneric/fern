#define BOOST_TEST_MODULE ranally language
#include <boost/test/included/unit_test.hpp>
#include "Ranally/Language/algebra_parser.h"
#include "Ranally/Language/assignment_vertex.h"
#include "Ranally/Language/identify_visitor.h"
#include "Ranally/Language/if_vertex.h"
#include "Ranally/Language/function_vertex.h"
#include "Ranally/Language/name_vertex.h"
#include "Ranally/Language/operator_vertex.h"
#include "Ranally/Language/script_vertex.h"
#include "Ranally/Language/xml_parser.h"


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

    ranally::IdentifyVisitor _visitor;
};


BOOST_FIXTURE_TEST_SUITE(identify_visitor, Support)

BOOST_AUTO_TEST_CASE(visit_empty_script)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String("")));
    tree->Accept(_visitor);
}


BOOST_AUTO_TEST_CASE(visit_name)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String("a")));

    ranally::NameVertex const* vertexA =
        dynamic_cast<ranally::NameVertex const*>(
            &(*tree->statements()[0]));

    BOOST_CHECK(vertexA->definitions().empty());
    BOOST_CHECK(vertexA->uses().empty());

    tree->Accept(_visitor);

    BOOST_CHECK(vertexA->definitions().empty());
    BOOST_CHECK(vertexA->uses().empty());
}


BOOST_AUTO_TEST_CASE(visit_assignment)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(
            ranally::String("a = b")));

        ranally::AssignmentVertex const* assignment =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertexA =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment->target()));
        ranally::NameVertex const* vertexB =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment->expression()));

        BOOST_CHECK(vertexA->definitions().empty());
        BOOST_CHECK(vertexA->uses().empty());

        BOOST_CHECK(vertexB->definitions().empty());
        BOOST_CHECK(vertexB->uses().empty());

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertexA->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA->definitions()[0], vertexA);
        BOOST_CHECK(vertexA->uses().empty());

        BOOST_CHECK(vertexB->definitions().empty());
        BOOST_CHECK(vertexB->uses().empty());
    }

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a = b\n"
            "d = f(a)\n"
        )));

        ranally::AssignmentVertex const* assignment1 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertexA1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment1->target()));
        ranally::NameVertex const* vertexB =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment1->expression()));

        ranally::AssignmentVertex const* assignment2 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[1]));
        ranally::FunctionVertex const* function =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment2->expression()));
        ranally::NameVertex const* vertexA2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*function->expressions()[0]));
        ranally::NameVertex const* vertexD =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment2->target()));

        BOOST_CHECK(vertexA1->definitions().empty());
        BOOST_CHECK(vertexA1->uses().empty());

        BOOST_CHECK(vertexB->definitions().empty());
        BOOST_CHECK(vertexB->uses().empty());

        BOOST_CHECK(vertexD->definitions().empty());
        BOOST_CHECK(vertexD->uses().empty());

        BOOST_CHECK(vertexA2->definitions().empty());
        BOOST_CHECK(vertexA2->uses().empty());

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertexA1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA1->definitions()[0], vertexA1);
        BOOST_REQUIRE_EQUAL(vertexA1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA1->uses()[0], vertexA2);

        BOOST_CHECK(vertexB->definitions().empty());
        BOOST_CHECK(vertexB->uses().empty());

        BOOST_REQUIRE_EQUAL(vertexD->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexD->definitions()[0], vertexD);
        BOOST_CHECK(vertexD->uses().empty());

        BOOST_REQUIRE_EQUAL(vertexA2->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA2->definitions()[0], vertexA1);
        BOOST_CHECK(vertexA2->uses().empty());
    }
}


BOOST_AUTO_TEST_CASE(visit_if)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a = b\n"
            "if True:\n"
            "    a = c\n"
            "d = a\n"
        )));

        ranally::AssignmentVertex const* assignment1 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertexA1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment1->target()));

        ranally::IfVertex const* ifVertex =
            dynamic_cast<ranally::IfVertex const*>(
                &(*tree->statements()[1]));
        ranally::AssignmentVertex const* assignment2 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*ifVertex->trueStatements()[0]));
        ranally::NameVertex const* vertexA2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment2->target()));

        ranally::AssignmentVertex const* assignment3 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[2]));
        ranally::NameVertex const* vertexA3 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment3->expression()));

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertexA1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA1->definitions()[0], vertexA1);
        BOOST_REQUIRE_EQUAL(vertexA1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA1->uses()[0], vertexA3);

        BOOST_REQUIRE_EQUAL(vertexA2->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA2->definitions()[0], vertexA2);
        // TODO
        // BOOST_REQUIRE_EQUAL(vertexA2->uses().size(), 1u);
        // BOOST_CHECK_EQUAL(vertexA2->uses()[0], vertexA3);

        // BOOST_REQUIRE_EQUAL(vertexA3->definitions().size(), 2u);
        // BOOST_CHECK_EQUAL(vertexA3->definitions()[0], vertexA1);
        // BOOST_CHECK_EQUAL(vertexA3->definitions()[0], vertexA2);
        // BOOST_CHECK(vertexA3->uses().empty());
    }
}


BOOST_AUTO_TEST_CASE(visit_reuse_of_identifiers)
{
    std::shared_ptr<ranally::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a = \"MyRaster\"\n"
            "b = abs(a)\n"
            "c = abs(b)\n"
            "b = c + b\n"
        )));

        ranally::AssignmentVertex const* assignment1 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::NameVertex const* vertexA1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment1->target()));
        BOOST_REQUIRE(assignment1);
        BOOST_REQUIRE(vertexA1);

        ranally::AssignmentVertex const* assignment2 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[1]));
        ranally::NameVertex const* vertexB1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment2->target()));
        ranally::FunctionVertex const* vertexAbs1 =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment2->expression()));
        ranally::NameVertex const* vertexA2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertexAbs1->expressions()[0]));
        BOOST_REQUIRE(assignment2);
        BOOST_REQUIRE(vertexB1);
        BOOST_REQUIRE(vertexAbs1);
        BOOST_REQUIRE(vertexA2);

        ranally::AssignmentVertex const* assignment3 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[2]));
        ranally::NameVertex const* vertexC1 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment3->target()));
        ranally::FunctionVertex const* vertexAbs2 =
            dynamic_cast<ranally::FunctionVertex const*>(
                &(*assignment3->expression()));
        ranally::NameVertex const* vertexB2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertexAbs2->expressions()[0]));
        BOOST_REQUIRE(assignment3);
        BOOST_REQUIRE(vertexC1);
        BOOST_REQUIRE(vertexAbs2);
        BOOST_REQUIRE(vertexB2);

        ranally::AssignmentVertex const* assignment4 =
            dynamic_cast<ranally::AssignmentVertex const*>(
                &(*tree->statements()[3]));
        ranally::NameVertex const* vertexB4 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*assignment4->target()));
        ranally::OperatorVertex const* vertexPlus1 =
            dynamic_cast<ranally::OperatorVertex const*>(
                &(*assignment4->expression()));
        ranally::NameVertex const* vertexC2 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertexPlus1->expressions()[0]));
        ranally::NameVertex const* vertexB3 =
            dynamic_cast<ranally::NameVertex const*>(
                &(*vertexPlus1->expressions()[1]));
        BOOST_REQUIRE(assignment4);
        BOOST_REQUIRE(vertexB4);
        BOOST_REQUIRE(vertexPlus1);
        BOOST_REQUIRE(vertexC2);
        BOOST_REQUIRE(vertexB3);

        tree->Accept(_visitor);

        BOOST_REQUIRE_EQUAL(vertexA1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA1->definitions()[0], vertexA1);
        BOOST_REQUIRE_EQUAL(vertexA1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertexA1->uses()[0], vertexA2);

        BOOST_REQUIRE_EQUAL(vertexB1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexB1->definitions()[0], vertexB1);
        BOOST_REQUIRE_EQUAL(vertexB1->uses().size(), 2u);
        BOOST_CHECK_EQUAL(vertexB1->uses()[0], vertexB2);
        BOOST_CHECK_EQUAL(vertexB1->uses()[1], vertexB3);

        BOOST_REQUIRE_EQUAL(vertexC1->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexC1->definitions()[0], vertexC1);
        BOOST_REQUIRE_EQUAL(vertexC1->uses().size(), 1u);
        BOOST_CHECK_EQUAL(vertexC1->uses()[0], vertexC2);

        BOOST_REQUIRE_EQUAL(vertexB4->definitions().size(), 1u);
        BOOST_CHECK_EQUAL(vertexB4->definitions()[0], vertexB4);
        BOOST_REQUIRE_EQUAL(vertexB4->uses().size(), 0u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
