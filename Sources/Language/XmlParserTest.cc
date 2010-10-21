#include "XmlParserTest.h"

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "dev_UnicodeUtils.h"

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
  boost::shared_ptr<ranally::SyntaxVertex> tree;

  {
    // Empty xml.
    xml = algebraParser.parseString(UnicodeString(""));
    tree = xmlParser.parse(xml);
  }

  {
    // Name expression.
    xml = algebraParser.parseString(UnicodeString("a"));
    tree = xmlParser.parse(xml);
  }

  {
    // String expression.
    xml = algebraParser.parseString(UnicodeString("\"five\""));
    tree = xmlParser.parse(xml);
  }

  {
    // Numeric expression.
    xml = algebraParser.parseString(UnicodeString("5"));
    tree = xmlParser.parse(xml);

    xml = algebraParser.parseString(UnicodeString("5L"));
    tree = xmlParser.parse(xml);

    xml = algebraParser.parseString(UnicodeString("5.5"));
    tree = xmlParser.parse(xml);
  }

  {
    // Function call.
    xml = algebraParser.parseString(UnicodeString("f()"));
    tree = xmlParser.parse(xml);

    xml = algebraParser.parseString(UnicodeString(
      "f(1, \"2\", three, four())"));
    tree = xmlParser.parse(xml);
  }

  {
    // Assignment statement.
    xml = algebraParser.parseString(UnicodeString("a = b"));
    tree = xmlParser.parse(xml);
  }

  {
    // If statement.
    xml = algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b"));
    tree = xmlParser.parse(xml);

    xml = algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "else:\n"
      "  c"));
    tree = xmlParser.parse(xml);

    xml = algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "elif c:\n"
      "  d"));
    tree = xmlParser.parse(xml);
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

