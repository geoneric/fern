#include "XmlParserTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

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
  ranally::XmlParser parser;

  // {
  //   ranally::Ast ast(parser.parse(UnicodeString xml(
  //     "<module>"
  //       "<expression line=\"1\" col=\"0\">"
  //         "<name>a</name>"
  //       "</expression>"
  //     "</module>")));
  // }

  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

