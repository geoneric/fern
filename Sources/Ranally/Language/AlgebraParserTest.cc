#include "Ranally/Language/AlgebraParserTest.h"
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Language/AlgebraParser.h"



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
    &AlgebraParserTest::testParseUnaryOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseBinaryOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseBooleanOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseComparisonOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseMultipleStatements, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseIf, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseWhile, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseFile, instance));

  return suite;
}



AlgebraParserTest::AlgebraParserTest()
{
}



void AlgebraParserTest::testParseEmptyScript()
{
  ranally::language::AlgebraParser parser;

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
  ranally::language::AlgebraParser parser;

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

  // TODO #if PYTHONVER >= 2.7/3.0?
  // {
  //   UnicodeString xml(parser.parseString(UnicodeString("ñ")));
  //   BOOST_CHECK(xml ==
  //     "<?xml version=\"1.0\"?>"
  //     "<Ranally>"
  //       "<Statements>"
  //         "<Statement>"
  //           "<Expression line=\"1\" col=\"0\">"
  //             "<Name>ñ</Name>"
  //           "</Expression>"
  //         "</Statement>"
  //       "</Statements>"
  //     "</Ranally>");
  // }
}



void AlgebraParserTest::testParseAssignment()
{
  ranally::language::AlgebraParser parser;

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
  ranally::language::AlgebraParser parser;

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
  {
    UnicodeString xml(parser.parseString(UnicodeString("\"mañana\"")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<String>mañana</String>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseNumber()
{
  ranally::language::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("5")));
    BOOST_CHECK(xml == (boost::format(
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Number>"
                "<Integer>"
                  "<Size>%1%</Size>"
                  "<Value>5</Value>"
                "</Integer>"
              "</Number>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>") % (sizeof(long) * 8)).str().c_str());
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
                "<Integer>"
                  "<Size>64</Size>"
                  "<Value>5</Value>"
                "</Integer>"
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
                "<Float>"
                  "<Size>64</Size>"
                  "<Value>5.5</Value>"
                "</Float>"
              "</Number>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseCall()
{
  ranally::language::AlgebraParser parser;
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
    BOOST_CHECK(xml == (boost::format(
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
                      "<Integer>"
                        "<Size>%1%</Size>"
                        "<Value>1</Value>"
                      "</Integer>"
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
      "</Ranally>") % (sizeof(long) * 8)).str().c_str());
  }
}



void AlgebraParserTest::testParseUnaryOperator()
{
  ranally::language::AlgebraParser parser;
  UnicodeString xml;

  {
    xml = parser.parseString(UnicodeString("-a"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Operator>"
                "<Name>Sub</Name>"
                "<Expressions>"
                  "<Expression line=\"1\" col=\"1\">"
                    "<Name>a</Name>"
                  "</Expression>"
                "</Expressions>"
              "</Operator>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseBinaryOperator()
{
  ranally::language::AlgebraParser parser;
  UnicodeString xml;

  {
    xml = parser.parseString(UnicodeString("a + b"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Operator>"
                "<Name>Add</Name>"
                "<Expressions>"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Name>a</Name>"
                  "</Expression>"
                  "<Expression line=\"1\" col=\"4\">"
                    "<Name>b</Name>"
                  "</Expression>"
                "</Expressions>"
              "</Operator>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseBooleanOperator()
{
  ranally::language::AlgebraParser parser;
  UnicodeString xml;

  {
    xml = parser.parseString(UnicodeString("a and b"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Operator>"
                "<Name>And</Name>"
                "<Expressions>"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Name>a</Name>"
                  "</Expression>"
                  "<Expression line=\"1\" col=\"6\">"
                    "<Name>b</Name>"
                  "</Expression>"
                "</Expressions>"
              "</Operator>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseComparisonOperator()
{
  ranally::language::AlgebraParser parser;
  UnicodeString xml;

  {
    xml = parser.parseString(UnicodeString("a <= b"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<Expression line=\"1\" col=\"0\">"
              "<Operator>"
                "<Name>LtE</Name>"
                "<Expressions>"
                  "<Expression line=\"1\" col=\"0\">"
                    "<Name>a</Name>"
                  "</Expression>"
                  "<Expression line=\"1\" col=\"5\">"
                    "<Name>b</Name>"
                  "</Expression>"
                "</Expressions>"
              "</Operator>"
            "</Expression>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseMultipleStatements()
{
  ranally::language::AlgebraParser parser;

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
  ranally::language::AlgebraParser parser;
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



void AlgebraParserTest::testParseWhile()
{
  ranally::language::AlgebraParser parser;
  UnicodeString xml;

  {
    xml = parser.parseString(UnicodeString(
      "while a:\n"
      "  b"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<While>"
              "<Expression line=\"1\" col=\"6\">"
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
            "</While>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }

  {
    xml = parser.parseString(UnicodeString(
      "while a:\n"
      "  b\n"
      "else:\n"
      "  c"));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Statements>"
          "<Statement>"
            "<While>"
              "<Expression line=\"1\" col=\"6\">"
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
                  "<Expression line=\"4\" col=\"2\">"
                    "<Name>c</Name>"
                  "</Expression>"
                "</Statement>"
              "</Statements>"
            "</While>"
          "</Statement>"
        "</Statements>"
      "</Ranally>");
  }
}



void AlgebraParserTest::testParseFile()
{
  ranally::language::AlgebraParser parser;
  UnicodeString fileName;

  {
    fileName = "DoesNotExist.ran";
    BOOST_CHECK_THROW(parser.parseFile(fileName),
      std::runtime_error);
  }
}

