#include "Ranally/Language/XmlParserTest.h"
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/Ranally-pskel.hxx"
#include "Ranally/Language/XmlParser.h"


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
    ranally::language::AlgebraParser algebraParser;
    ranally::language::XmlParser xmlParser;
    ranally::String xml;
    boost::shared_ptr<ranally::language::SyntaxVertex> tree;

    {
        // Empty xml.
        xml = algebraParser.parseString(ranally::String(""));
        tree = xmlParser.parse(xml);
    }

    {
        // Name expression.
        xml = algebraParser.parseString(ranally::String("a"));
        tree = xmlParser.parse(xml);
    }

    {
        // String expression.
        xml = algebraParser.parseString(ranally::String("\"five\""));
        tree = xmlParser.parse(xml);
    }

    {
        // TODO test all numeric types.

        // Numeric expression.
        xml = algebraParser.parseString(ranally::String("5"));
        tree = xmlParser.parse(xml);

        xml = algebraParser.parseString(ranally::String("5L"));
        tree = xmlParser.parse(xml);

        // TODO test unsigned types.

        xml = algebraParser.parseString(ranally::String("5.5"));
        tree = xmlParser.parse(xml);
    }

    {
        // Function call.
        xml = algebraParser.parseString(ranally::String("f()"));
        tree = xmlParser.parse(xml);

        xml = algebraParser.parseString(ranally::String(
            "f(1, \"2\", three, four())"));
        tree = xmlParser.parse(xml);
    }

    {
        // Operator.
        // Unary.
        xml = algebraParser.parseString(ranally::String("-a"));
        tree = xmlParser.parse(xml);

        // Binary.
        xml = algebraParser.parseString(ranally::String("a + b"));
        tree = xmlParser.parse(xml);

        // Boolean.
        xml = algebraParser.parseString(ranally::String("a and b"));
        tree = xmlParser.parse(xml);

        // Comparison.
        xml = algebraParser.parseString(ranally::String("a == b"));
        tree = xmlParser.parse(xml);
    }

    {
        // Assignment statement.
        xml = algebraParser.parseString(ranally::String("a = b"));
        tree = xmlParser.parse(xml);
    }

    {
        // Multiple statements.
        xml = algebraParser.parseString(ranally::String("a\nb"));
        tree = xmlParser.parse(xml);
    }

    {
        // If statement.
        xml = algebraParser.parseString(ranally::String(
            "if a:\n"
            "    b"));
        tree = xmlParser.parse(xml);

        xml = algebraParser.parseString(ranally::String(
            "if a:\n"
            "    b\n"
            "else:\n"
            "    c"));
        tree = xmlParser.parse(xml);

        xml = algebraParser.parseString(ranally::String(
            "if a:\n"
            "    b\n"
            "elif c:\n"
            "    d"));
        tree = xmlParser.parse(xml);
    }

    {
        // While statement.
        xml = algebraParser.parseString(ranally::String(
            "while a:\n"
            "    b"));
        tree = xmlParser.parse(xml);

        xml = algebraParser.parseString(ranally::String(
            "while a:\n"
            "    b\n"
            "else:\n"
            "    c"));
        tree = xmlParser.parse(xml);
    }

    {
        // Random string.
        BOOST_CHECK_THROW(xmlParser.parse(ranally::String("blabla")),
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
