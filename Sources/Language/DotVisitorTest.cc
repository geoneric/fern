#include "DotVisitorTest.h"

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

// #include "dev_UnicodeUtils.h"

#include "ScriptVertex.h"



boost::unit_test::test_suite* DotVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<DotVisitorTest> instance(
    new DotVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitEmptyScript, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitName, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitAssignment, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitString, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitNumber, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitFunction, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitMultipleStatements, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitIf, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &DotVisitorTest::testVisitWhile, instance));

  return suite;
}



DotVisitorTest::DotVisitorTest()
{
}



void DotVisitorTest::testVisitEmptyScript()
{
  UnicodeString xml;

  xml = _algebraParser.parseString(UnicodeString(""));
  BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "");
}



void DotVisitorTest::testVisitName()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString("a"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a\n");
}



void DotVisitorTest::testVisitAssignment()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString("a = b"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a = b\n");
}



void DotVisitorTest::testVisitString()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString("\"five\""));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "\"five\"\n");
}



void DotVisitorTest::testVisitNumber()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString("5"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5\n");

  // xml = _algebraParser.parseString(UnicodeString("5L"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5L\n");

  // xml = _algebraParser.parseString(UnicodeString("5.5"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "5.5\n");

  // // TODO add tests for all numeric types.
}



void DotVisitorTest::testVisitFunction()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString("f()"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "f()\n");

  // xml = _algebraParser.parseString(UnicodeString("f(1, \"2\", three, four())"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "f(1, \"2\", three, four())\n");
}



void DotVisitorTest::testVisitOperator()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString("-a"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-(a)\n");

  // xml = _algebraParser.parseString(UnicodeString("a + b"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "(a) + (b)\n");

  // xml = _algebraParser.parseString(UnicodeString("-(a + b)"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-((a) + (b))\n");

  // xml = _algebraParser.parseString(UnicodeString("a + b * c + d"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "((a) + ((b) * (c))) + (d)\n");
}



void DotVisitorTest::testVisitMultipleStatements()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString("a\nb"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a\nb\n");
}



void DotVisitorTest::testVisitIf()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString(
  //     "if a:\n"
  //     "  b\n"
  //     "  c"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //     "if a:\n"
  //     "  b\n"
  //     "  c\n");

  // xml = _algebraParser.parseString(UnicodeString(
  //     "if a:\n"
  //     "  b\n"
  //     "  c\n"
  //     "elif d:\n"
  //     "  e\n"
  //     "  f\n"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //     "if a:\n"
  //     "  b\n"
  //     "  c\n"
  //     "else:\n"
  //     "  if d:\n"
  //     "    e\n"
  //     "    f\n");

  // xml = _algebraParser.parseString(UnicodeString(
  //     "if a:\n"
  //     "  b\n"
  //     "  c\n"
  //     "elif d:\n"
  //     "  e\n"
  //     "  f\n"
  //     "else:\n"
  //     "  g\n"
  //     "  h\n"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //     "if a:\n"
  //     "  b\n"
  //     "  c\n"
  //     "else:\n"
  //     "  if d:\n"
  //     "    e\n"
  //     "    f\n"
  //     "  else:\n"
  //     "    g\n"
  //     "    h\n");
}



void DotVisitorTest::testVisitWhile()
{
  // UnicodeString xml;

  // xml = _algebraParser.parseString(UnicodeString(
  //     "while a:\n"
  //     "  b\n"
  //     "  c"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //     "while a:\n"
  //     "  b\n"
  //     "  c\n");

  // xml = _algebraParser.parseString(UnicodeString(
  //     "while a:\n"
  //     "  b\n"
  //     "  c\n"
  //     "else:\n"
  //     "  d\n"
  //     "  e"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //     "while a:\n"
  //     "  b\n"
  //     "  c\n"
  //     "else:\n"
  //     "  d\n"
  //     "  e\n");
}
