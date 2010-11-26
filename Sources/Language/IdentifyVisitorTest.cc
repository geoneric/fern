#include "IdentifyVisitorTest.h"

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "AssignmentVertex.h"
#include "FunctionVertex.h"
#include "NameVertex.h"
#include "ScriptVertex.h"



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

  return suite;
}



IdentifyVisitorTest::IdentifyVisitorTest()
{
}



void IdentifyVisitorTest::testVisitEmptyScript()
{
  boost::shared_ptr<ranally::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("")));
  tree->Accept(_visitor);
}



void IdentifyVisitorTest::testVisitName()
{
  boost::shared_ptr<ranally::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("a")));

  ranally::NameVertex const* vertexA =
    dynamic_cast<ranally::NameVertex const*>(&(*tree->statements()[0]));

  BOOST_CHECK(!vertexA->definition());
  BOOST_CHECK(vertexA->uses().empty());

  tree->Accept(_visitor);

  BOOST_CHECK(!vertexA->definition());
  BOOST_CHECK(vertexA->uses().empty());
}



void IdentifyVisitorTest::testVisitAssignment()
{
  boost::shared_ptr<ranally::ScriptVertex> tree;

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("a = b")));

    ranally::AssignmentVertex const* assignment =
      dynamic_cast<ranally::AssignmentVertex const*>(&(*tree->statements()[0]));
    ranally::NameVertex const* vertexA =
      dynamic_cast<ranally::NameVertex const*>(&(*assignment->targets()[0]));
    ranally::NameVertex const* vertexB =
      dynamic_cast<ranally::NameVertex const*>(&(*assignment->expressions()[0]));

    BOOST_CHECK(!vertexA->definition());
    BOOST_CHECK(vertexA->uses().empty());

    BOOST_CHECK(!vertexB->definition());
    BOOST_CHECK(vertexB->uses().empty());

    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(vertexA->definition(), vertexA);
    BOOST_CHECK(vertexA->uses().empty());

    BOOST_CHECK(!vertexB->definition());
    BOOST_CHECK(vertexB->uses().empty());
  }

  {
    tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString(
      "a = b\n"
      "d = f(a)\n"
    )));

    ranally::AssignmentVertex const* assignment1 =
      dynamic_cast<ranally::AssignmentVertex const*>(&(*tree->statements()[0]));
    ranally::NameVertex const* vertexA1 =
      dynamic_cast<ranally::NameVertex const*>(&(*assignment1->targets()[0]));
    ranally::NameVertex const* vertexB =
      dynamic_cast<ranally::NameVertex const*>(&(*assignment1->expressions()[0]));

    ranally::AssignmentVertex const* assignment2 =
      dynamic_cast<ranally::AssignmentVertex const*>(&(*tree->statements()[1]));
    ranally::FunctionVertex const* function =
      dynamic_cast<ranally::FunctionVertex const*>(&(*assignment2->expressions()[0]));
    ranally::NameVertex const* vertexA2 =
      dynamic_cast<ranally::NameVertex const*>(&(*function->expressions()[0]));
    ranally::NameVertex const* vertexD =
      dynamic_cast<ranally::NameVertex const*>(&(*assignment2->targets()[0]));

    BOOST_CHECK(!vertexA1->definition());
    BOOST_CHECK(vertexA1->uses().empty());

    BOOST_CHECK(!vertexB->definition());
    BOOST_CHECK(vertexB->uses().empty());

    BOOST_CHECK(!vertexD->definition());
    BOOST_CHECK(vertexD->uses().empty());

    BOOST_CHECK(!vertexA2->definition());
    BOOST_CHECK(vertexA2->uses().empty());

    tree->Accept(_visitor);

    BOOST_CHECK_EQUAL(vertexA1->definition(), vertexA1);
    BOOST_REQUIRE_EQUAL(vertexA1->uses().size(), 1);
    BOOST_CHECK_EQUAL(vertexA1->uses()[0], vertexA2);

    BOOST_CHECK(!vertexB->definition());
    BOOST_CHECK(vertexB->uses().empty());

    BOOST_CHECK_EQUAL(vertexD->definition(), vertexD);
    BOOST_CHECK(vertexD->uses().empty());

    BOOST_CHECK_EQUAL(vertexA2->definition(), vertexA1);
    BOOST_CHECK(vertexA2->uses().empty());
  }
}

