#include "Ranally/Language/ScriptVisitorTest.h"

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Language/ScriptVertex.h"



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
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "");
}



void ScriptVisitorTest::testVisitName()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("a"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "a\n");
}



void ScriptVisitorTest::testVisitAssignment()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("a = b"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "a = b\n");
}



void ScriptVisitorTest::testVisitString()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("\"five\""));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "\"five\"\n");
}



void ScriptVisitorTest::testVisitNumber()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("5"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "5\n");

  xml = _algebraParser.parseString(UnicodeString("5L"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "5L\n");

  xml = _algebraParser.parseString(UnicodeString("5.5"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "5.5\n");

  // TODO add tests for all numeric types.
}



void ScriptVisitorTest::testVisitFunction()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("f()"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "f()\n");

  xml = _algebraParser.parseString(UnicodeString("f(1, \"2\", three, four())"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() ==
    "f(1, \"2\", three, four())\n");
}



void ScriptVisitorTest::testVisitOperator()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("-a"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "-(a)\n");

  xml = _algebraParser.parseString(UnicodeString("a + b"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "(a) + (b)\n");

  xml = _algebraParser.parseString(UnicodeString("-(a + b)"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "-((a) + (b))\n");

  xml = _algebraParser.parseString(UnicodeString("a + b * c + d"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() ==
    "((a) + ((b) * (c))) + (d)\n");
}



void ScriptVisitorTest::testVisitMultipleStatements()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString("a\nb"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() == "a\nb\n");
}



void ScriptVisitorTest::testVisitIf()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString(
      "if a:\n"
      "  b\n"
      "  c"));
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() ==
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
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() ==
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
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() ==
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
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() ==
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
  _xmlParser.parse(xml)->Accept(_visitor);
  BOOST_CHECK(_visitor.script() ==
      "while a:\n"
      "  b\n"
      "  c\n"
      "else:\n"
      "  d\n"
      "  e\n");
}
