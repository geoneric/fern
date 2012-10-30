#include "Ranally/Language/IdentifyVisitorTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Language/AssignmentVertex.h"
#include "Ranally/Language/IfVertex.h"
#include "Ranally/Language/FunctionVertex.h"
#include "Ranally/Language/NameVertex.h"
#include "Ranally/Language/OperatorVertex.h"
#include "Ranally/Language/ScriptVertex.h"


boost::unit_test::test_suite* IdentifyVisitorTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<IdentifyVisitorTest> instance(
        new IdentifyVisitorTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &IdentifyVisitorTest::testVisitEmptyScript, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &IdentifyVisitorTest::testVisitName, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &IdentifyVisitorTest::testVisitAssignment, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &IdentifyVisitorTest::testVisitIf, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &IdentifyVisitorTest::testVisitReuseOfIdentifiers, instance));

    return suite;
}


IdentifyVisitorTest::IdentifyVisitorTest()
{
}


void IdentifyVisitorTest::testVisitEmptyScript()
{
    boost::shared_ptr<ranally::language::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String("")));
    tree->Accept(_visitor);
}


void IdentifyVisitorTest::testVisitName()
{
    boost::shared_ptr<ranally::language::ScriptVertex> tree;

    tree = _xmlParser.parse(_algebraParser.parseString(ranally::String("a")));

    ranally::language::NameVertex const* vertexA =
        dynamic_cast<ranally::language::NameVertex const*>(
            &(*tree->statements()[0]));

    BOOST_CHECK(vertexA->definitions().empty());
    BOOST_CHECK(vertexA->uses().empty());

    tree->Accept(_visitor);

    BOOST_CHECK(vertexA->definitions().empty());
    BOOST_CHECK(vertexA->uses().empty());
}


void IdentifyVisitorTest::testVisitAssignment()
{
    boost::shared_ptr<ranally::language::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(
            ranally::String("a = b")));

        ranally::language::AssignmentVertex const* assignment =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::language::NameVertex const* vertexA =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment->target()));
        ranally::language::NameVertex const* vertexB =
            dynamic_cast<ranally::language::NameVertex const*>(
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

        ranally::language::AssignmentVertex const* assignment1 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::language::NameVertex const* vertexA1 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment1->target()));
        ranally::language::NameVertex const* vertexB =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment1->expression()));

        ranally::language::AssignmentVertex const* assignment2 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[1]));
        ranally::language::FunctionVertex const* function =
            dynamic_cast<ranally::language::FunctionVertex const*>(
                &(*assignment2->expression()));
        ranally::language::NameVertex const* vertexA2 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*function->expressions()[0]));
        ranally::language::NameVertex const* vertexD =
            dynamic_cast<ranally::language::NameVertex const*>(
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


void IdentifyVisitorTest::testVisitIf()
{
    boost::shared_ptr<ranally::language::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a = b\n"
            "if True:\n"
            "    a = c\n"
            "d = a\n"
        )));

        ranally::language::AssignmentVertex const* assignment1 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::language::NameVertex const* vertexA1 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment1->target()));

        ranally::language::IfVertex const* ifVertex =
            dynamic_cast<ranally::language::IfVertex const*>(
                &(*tree->statements()[1]));
        ranally::language::AssignmentVertex const* assignment2 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*ifVertex->trueStatements()[0]));
        ranally::language::NameVertex const* vertexA2 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment2->target()));

        ranally::language::AssignmentVertex const* assignment3 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[2]));
        ranally::language::NameVertex const* vertexA3 =
            dynamic_cast<ranally::language::NameVertex const*>(
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


void IdentifyVisitorTest::testVisitReuseOfIdentifiers()
{
    boost::shared_ptr<ranally::language::ScriptVertex> tree;

    {
        tree = _xmlParser.parse(_algebraParser.parseString(ranally::String(
            "a = \"MyRaster\"\n"
            "b = abs(a)\n"
            "c = abs(b)\n"
            "b = c + b\n"
        )));

        ranally::language::AssignmentVertex const* assignment1 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[0]));
        ranally::language::NameVertex const* vertexA1 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment1->target()));
        BOOST_REQUIRE(assignment1);
        BOOST_REQUIRE(vertexA1);

        ranally::language::AssignmentVertex const* assignment2 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[1]));
        ranally::language::NameVertex const* vertexB1 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment2->target()));
        ranally::language::FunctionVertex const* vertexAbs1 =
            dynamic_cast<ranally::language::FunctionVertex const*>(
                &(*assignment2->expression()));
        ranally::language::NameVertex const* vertexA2 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*vertexAbs1->expressions()[0]));
        BOOST_REQUIRE(assignment2);
        BOOST_REQUIRE(vertexB1);
        BOOST_REQUIRE(vertexAbs1);
        BOOST_REQUIRE(vertexA2);

        ranally::language::AssignmentVertex const* assignment3 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[2]));
        ranally::language::NameVertex const* vertexC1 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment3->target()));
        ranally::language::FunctionVertex const* vertexAbs2 =
            dynamic_cast<ranally::language::FunctionVertex const*>(
                &(*assignment3->expression()));
        ranally::language::NameVertex const* vertexB2 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*vertexAbs2->expressions()[0]));
        BOOST_REQUIRE(assignment3);
        BOOST_REQUIRE(vertexC1);
        BOOST_REQUIRE(vertexAbs2);
        BOOST_REQUIRE(vertexB2);

        ranally::language::AssignmentVertex const* assignment4 =
            dynamic_cast<ranally::language::AssignmentVertex const*>(
                &(*tree->statements()[3]));
        ranally::language::NameVertex const* vertexB4 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*assignment4->target()));
        ranally::language::OperatorVertex const* vertexPlus1 =
            dynamic_cast<ranally::language::OperatorVertex const*>(
                &(*assignment4->expression()));
        ranally::language::NameVertex const* vertexC2 =
            dynamic_cast<ranally::language::NameVertex const*>(
                &(*vertexPlus1->expressions()[0]));
        ranally::language::NameVertex const* vertexB3 =
            dynamic_cast<ranally::language::NameVertex const*>(
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
