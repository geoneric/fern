#include "XmlParserTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "AlgebraParser.h"
#include "Ranally-pskel.hxx"
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
  UnicodeString xml;

  {
    // Empty xml.
    xml = algebraParser.parseString(UnicodeString(""));
    boost::shared_ptr<ranally::SyntaxTree> tree(xmlParser.parse(xml));
  }

  {
    // Name expression.
    xml = algebraParser.parseString(UnicodeString("a"));
    boost::shared_ptr<ranally::SyntaxTree> tree(xmlParser.parse(xml));
  }

  {
    // Assignment statement.
    xml = algebraParser.parseString(UnicodeString("a = b"));
    boost::shared_ptr<ranally::SyntaxTree> tree(xmlParser.parse(xml));
  }

  {
    // Random string.
    BOOST_CHECK_THROW(xmlParser.parse(UnicodeString("blabla")),
      xml_schema::parsing);

    // Attribute value missing.
    xml =
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Expression col=\"0\">"
          "<Name>a</Name>"
        "</Expression>"
      "</Ranally>";
    BOOST_CHECK_THROW(xmlParser.parse(xml), xml_schema::parsing);

    // Attribute value out of range.
    xml =
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Expression line=\"-1\" col=\"0\">"
          "<Name>a</Name>"
        "</Expression>"
      "</Ranally>";
    BOOST_CHECK_THROW(xmlParser.parse(xml), xml_schema::parsing);
  }
}

