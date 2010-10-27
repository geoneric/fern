#include "AlgebraParserTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "dev_UnicodeUtils.h"

#include "AlgebraParser.h"



boost::unit_test::test_suite* AlgebraParserTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<AlgebraParserTest> instance(
    new AlgebraParserTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseEmptyScript, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseName, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseAssignment, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseString, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseNumber, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseCall, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseMultipleStatements, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseIf, instance));

  /// suite->add(BOOST_CLASS_TEST_CASE(
  ///   &AlgebraParserTest::testParseFile, instance));

  return suite;
}



AlgebraParserTest::AlgebraParserTest()
{
}



void AlgebraParserTest::testParseEmptyScript()
{
  ranally::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements/>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseName()
{
  ranally::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("a")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Name>a</Name>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseAssignment()
{
  ranally::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("a = b")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Assignment>"
              "<Targets>"
                "<Expression line=\"1\" col=\"0\">"
                  "<Name>a</Name>"
                "</Expression>"
              "</Targets>"
              "<Expressions>"
                "<Expression line=\"1\" col=\"4\">"
                  "<Name>b</Name>"
                "</Expression>"
              "</Expressions>"
            "</Assignment>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseString()
{
  ranally::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("\"five\"")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<String>five</String>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    UnicodeString xml(parser.parseString(UnicodeString("\"\"")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<String/>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    UnicodeString xml(parser.parseString(UnicodeString("\" \"")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<String> </String>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  // Test handling of Unicode characters.
}



void AlgebraParserTest::testParseNumber()
{
  ranally::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("5")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Number>"
                "<Integer>5</Integer>"
              "</Number>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    UnicodeString xml(parser.parseString(UnicodeString("5L")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Number>"
                "<Long>5</Long>"
              "</Number>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    UnicodeString xml(parser.parseString(UnicodeString("5.5")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Number>"
                "<Double>5.5</Double>"
              "</Number>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseCall()
{
  ranally::AlgebraParser parser;
  UnicodeString xml;

  {
    xml = parser.parseString(UnicodeString("f()"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Function>"
                "<Name>f</Name>"
                "<Expressions/>"
              "</Function>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    xml = parser.parseString(UnicodeString("f(1, \"2\", three, four())"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Function>"
                "<Name>f</Name>"
                "<Expressions>"
                  "<Expression line=\"1\" col=\"2\">"
                    "<Number>"
                      "<Integer>1</Integer>"
                    "</Number>"
                  "</Expression>"
                  "<Expression line=\"1\" col=\"5\">"
                    "<String>2</String>"
                  "</Expression>"
                  "<Expression line=\"1\" col=\"10\">"
                    "<Name>three</Name>"
                  "</Expression>"
                  "<Expression line=\"1\" col=\"17\">"
                    "<Function>"
                      "<Name>four</Name>"
                      "<Expressions/>"
                    "</Function>"
                  "</Expression>"
                "</Expressions>"
              "</Function>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseMultipleStatements()
{
  ranally::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("a\nb")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Name>a</Name>"
            "</Expression>"
          "</Statement>"
          "<Statement>"
            "<Expression line=\"2\" col=\"0\">"
              "<Name>b</Name>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseIf()
{
  ranally::AlgebraParser parser;
  UnicodeString xml;

  {
    xml = parser.parseString(UnicodeString(
      "if a:\n"
      "  b"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<If>"
              "<Expression line=\"1\" col=\"3\">"
                "<Name>a</Name>"
              "</Expression>"
              "<Statements>"
                "<Statement>"
                  "<Expression line=\"2\" col=\"2\">"
                    "<Name>b</Name>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
              "<Statements/>"
            "</If>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    xml = parser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "elif(c):\n"
      "  d"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<If>"
              "<Expression line=\"1\" col=\"3\">"
                "<Name>a</Name>"
              "</Expression>"
              "<Statements>"
                "<Statement>"
                  "<Expression line=\"2\" col=\"2\">"
                    "<Name>b</Name>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
              "<Statements>"
                "<Statement>"
                  "<If>"
                    "<Expression line=\"3\" col=\"5\">"
                      "<Name>c</Name>"
                    "</Expression>"
                    "<Statements>"
                      "<Statement>"
                        "<Expression line=\"4\" col=\"2\">"
                          "<Name>d</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                    "<Statements/>"
                  "</If>"
                "</Statement>"
              "</Statements>"
            "</If>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    xml = parser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "elif c:\n"
      "  d\n"
      "else:\n"
      "  e"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<If>"
              "<Expression line=\"1\" col=\"3\">"
                "<Name>a</Name>"
              "</Expression>"
              "<Statements>"
                "<Statement>"
                  "<Expression line=\"2\" col=\"2\">"
                    "<Name>b</Name>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
              "<Statements>"
                "<Statement>"
                  "<If>"
                    "<Expression line=\"3\" col=\"5\">"
                      "<Name>c</Name>"
                    "</Expression>"
                    "<Statements>"
                      "<Statement>"
                        "<Expression line=\"4\" col=\"2\">"
                          "<Name>d</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                    "<Statements>"
                      "<Statement>"
                        "<Expression line=\"6\" col=\"2\">"
                          "<Name>e</Name>"
                        "</Expression>"
                      "</Statement>"
                    "</Statements>"
                  "</If>"
                "</Statement>"
              "</Statements>"
            "</If>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



/// void AlgebraParserTest::testParseFile()
/// {
///   bool testImplemented = false;
///   BOOST_WARN(testImplemented);
/// }

