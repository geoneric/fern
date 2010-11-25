#include "IdentifyVisitorTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "AssignmentVertex.h"
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



void IdentifyVisitorTest::testVisitAssignment()
{
  boost::shared_ptr<ranally::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("a = b")));

  ranally::AssignmentVertex const* assignment =
    dynamic_cast<ranally::AssignmentVertex const*>(&(*tree->statements()[0]));
  ranally::NameVertex const* vertexA =
    dynamic_cast<ranally::NameVertex const*>(&(*assignment->targets()[0]));
  ranally::NameVertex const* vertexB =
    dynamic_cast<ranally::NameVertex const*>(&(*assignment->expressions()[0]));

  BOOST_CHECK(!vertexA->definition());
  BOOST_CHECK(!vertexB->definition());

  tree->Accept(_visitor);

  BOOST_CHECK_EQUAL(vertexA->definition(), vertexA);
  BOOST_CHECK(!vertexB->definition());
}

