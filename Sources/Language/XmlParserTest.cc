#include "XmlParserTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "AlgebraParser.h"
#include "XmlParser.h"



boost::unit_test::test_suite* XmlParserTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<XmlParserTest> instance(
    new XmlParserTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &XmlParserTest::testParse, instance));

  return suite;
}



XmlParserTest::XmlParserTest()
{
}



void XmlParserTest::testParse()
{
  ranally::AlgebraParser algebraParser;
  ranally::XmlParser xmlParser;

  {
    UnicodeString xml(algebraParser.parseString(UnicodeString("a")));
    ranally::SyntaxTree tree(xmlParser.parse(xml));
  }
}

