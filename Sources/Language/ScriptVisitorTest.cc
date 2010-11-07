#include "ScriptVisitorTest.h"

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "dev_UnicodeUtils.h"

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
    &ScriptVisitorTest::testVisitFunction, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitMultipleStatements, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitIf, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ScriptVisitorTest::testVisitWhile, instance));

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
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a\n");
}



void ScriptVisitorTest::testVisitAssignment()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("a = b"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a = b\n");
}



void ScriptVisitorTest::testVisitString()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("\"five\""));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "\"five\"\n");
}



void ScriptVisitorTest::testVisitNumber()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("5"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5\n");

  xml = _algebraParser.parseString(UnicodeString("5L"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5L\n");

  xml = _algebraParser.parseString(UnicodeString("5.5"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5.5\n");
}



void ScriptVisitorTest::testVisitFunction()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("f()"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "f()\n");

  xml = _algebraParser.parseString(UnicodeString("f(1, \"2\", three, four())"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    "f(1, \"2\", three, four())\n");
}



void ScriptVisitorTest::testVisitOperator()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("-a"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-(a)\n");

  xml = _algebraParser.parseString(UnicodeString("a + b"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "(a) + (b)\n");

  xml = _algebraParser.parseString(UnicodeString("-(a + b)"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-((a) + (b))\n");

  xml = _algebraParser.parseString(UnicodeString("a + b * c + d"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
    "((a) + ((b) * (c))) + (d)\n");
}



void ScriptVisitorTest::testVisitMultipleStatements()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("a\nb"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a\nb\n");
}



void ScriptVisitorTest::testVisitIf()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "  c"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
      "if a:\n"
      "  b\n"
      "  c\n");

  xml = _algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "  c\n"
      "elif d:\n"
      "  e\n"
      "  f\n"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
      "if a:\n"
      "  b\n"
      "  c\n"
      "else:\n"
      "  if d:\n"
      "    e\n"
      "    f\n");

  xml = _algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "  c\n"
      "elif d:\n"
      "  e\n"
      "  f\n"
      "else:\n"
      "  g\n"
      "  h\n"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
      "if a:\n"
      "  b\n"
      "  c\n"
      "else:\n"
      "  if d:\n"
      "    e\n"
      "    f\n"
      "  else:\n"
      "    g\n"
      "    h\n");
}



void ScriptVisitorTest::testVisitWhile()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString(
      "while a:\n"
      "  b\n"
      "  c"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
      "while a:\n"
      "  b\n"
      "  c\n");

  xml = _algebraParser.parseString(UnicodeString(
      "while a:\n"
      "  b\n"
      "  c\n"
      "else:\n"
      "  d\n"
      "  e"));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
      "while a:\n"
      "  b\n"
      "  c\n"
      "else:\n"
      "  d\n"
      "  e\n");
}
