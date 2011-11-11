#include "Ranally/Language/AnnotateVisitorTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Language/ScriptVertex.h"



boost::unit_test::test_suite* AnnotateVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<AnnotateVisitorTest> instance(
    new AnnotateVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &AnnotateVisitorTest::testVisitEmptyScript, instance));

  return suite;
}



AnnotateVisitorTest::AnnotateVisitorTest()
{
}



void AnnotateVisitorTest::testVisitEmptyScript()
{
  // Parse empty script.
  // Ast before and after should be the same.
  boost::shared_ptr<ranally::language::ScriptVertex> tree1, tree2;

  tree1 = _xmlParser.parse(_algebraParser.parseString(UnicodeString("")));
  tree2.reset(new ranally::language::ScriptVertex(*tree1));
  tree1->Accept(_visitor);
  BOOST_CHECK(*tree1 == *tree2);
}

