#include "ScriptVisitorTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "AlgebraParser.h"
#include "ScriptVisitor.h"
#include "SyntaxTree.h"
#include "XmlParser.h"



boost::unit_test::test_suite* ScriptVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<ScriptVisitorTest> instance(
    new ScriptVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::test, instance));

  return suite;
}



ScriptVisitorTest::ScriptVisitorTest()
{
}



void ScriptVisitorTest::test()
{
  ranally::AlgebraParser algebraParser;
  ranally::XmlParser xmlParser;
  UnicodeString xml;
  boost::shared_ptr<ranally::SyntaxTree> tree;
  ranally::ScriptVisitor visitor;

  {
    xml = algebraParser.parseString(UnicodeString("a"));
    tree = xmlParser.parse(xml);
    tree->Accept(visitor);
  }
}

