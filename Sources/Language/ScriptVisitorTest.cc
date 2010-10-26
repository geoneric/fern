#include "ScriptVisitorTest.h"

// #include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

// #include "dev_UnicodeUtils.h"

#include "ScriptVertex.h"



boost::unit_test::test_suite* ScriptVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<ScriptVisitorTest> instance(
    new ScriptVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitEmptyScript, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitName, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitAssignment, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitString, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitNumber, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitCall, instance));

  return suite;
}



ScriptVisitorTest::ScriptVisitorTest()
{
}



void ScriptVisitorTest::testVisitEmptyScript()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString(""));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "");
}



void ScriptVisitorTest::testVisitName()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("a"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a");
}



void ScriptVisitorTest::testVisitAssignment()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("a = b"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a = b");
}



void ScriptVisitorTest::testVisitString()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("\"five\""));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "\"five\"");
}



void ScriptVisitorTest::testVisitNumber()
{
  UnicodeString xml;

  // TODO hier verder
  // xml = _algebraParser.parseString(UnicodeString("5"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5");

  // xml = _algebraParser.parseString(UnicodeString("5L"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5L");

  // xml = _algebraParser.parseString(UnicodeString("5.5"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5.5");
}



void ScriptVisitorTest::testVisitCall()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("f()"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "f()");

  xml = _algebraParser.parseString(UnicodeString("f(1, \"2\", three, four())"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    "f(1, \"2\", three, four())");
}


