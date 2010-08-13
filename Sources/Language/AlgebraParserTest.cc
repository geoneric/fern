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
         &AlgebraParserTest::testParseString, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &AlgebraParserTest::testParseFile, instance));

  return suite;
}



AlgebraParserTest::AlgebraParserTest()
{
}



void AlgebraParserTest::testParseString()
{
  ranally::AlgebraParser parser;
  UnicodeString xml(parser.parseString(UnicodeString("a")));

  std::cout << dev::encodeInUTF8(xml) << std::endl;

  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}



void AlgebraParserTest::testParseFile()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

