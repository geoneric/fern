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
    &AlgebraParserTest::testParseNameExpression, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &AlgebraParserTest::testParseAssignment, instance));



  /// suite->add(BOOST_CLASS_TEST_CASE(
  ///   &AlgebraParserTest::testParseFile, instance));

  return suite;
}



AlgebraParserTest::AlgebraParserTest()
{
}



void AlgebraParserTest::testParseNameExpression()
{
  ranally::AlgebraParser parser;

  {
    UnicodeString xml(parser.parseString(UnicodeString("a")));
    BOOST_CHECK(xml ==
      "<?xml version=\"1.0\"?>"
      "<Ranally>"
        "<Expression line=\"1\" col=\"0\">"
          "<Name>a</Name>"
        "</Expression>"
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
        "</Ranally>");
  }
}



/// void AlgebraParserTest::testParseFile()
/// {
///   bool testImplemented = false;
///   BOOST_WARN(testImplemented);
/// }

