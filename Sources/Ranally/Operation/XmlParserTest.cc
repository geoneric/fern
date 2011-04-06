#include "XmlParserTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Operation/XmlParser.h"



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
  ranally::operation::XmlParser xmlParser;
  UnicodeString xml;
  std::map<UnicodeString, ranally::operation::Operation_pskel> operations;

  {
    // Empty xml.
    xml =
      "<?xml version=\"1.0\"?>"
      "<Operations/>";
    operations = xmlParser.parse(xml);
    BOOST_CHECK(operations.empty());
  }

}

