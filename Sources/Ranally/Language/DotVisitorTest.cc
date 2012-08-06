#include "Ranally/Language/DotVisitorTest.h"
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/Language/ScriptVertex.h"



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
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String(""));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "}\n"
  // );
}



void DotVisitorTest::testVisitName()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String("a"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  a;\n"
  //   "}\n"
  // );
}



void DotVisitorTest::testVisitAssignment()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String("a = b"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  b -> a;\n"
  //   "}\n"
  // );
}



void DotVisitorTest::testVisitString()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String("\"five\""));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  \"five\";\n"
  //   "}\n"
  // );
}



void DotVisitorTest::testVisitNumber()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String("5"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  5;\n"
  //   "}\n"
  // );

  // xml = _algebraParser.parseString(ranally::String("5L"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  5;\n"
  //   "}\n"
  // );

  // xml = _algebraParser.parseString(ranally::String("5.5"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  5.5;\n"
  //   "}\n"
  // );

  // // TODO add tests for all numeric types.
}



void DotVisitorTest::testVisitFunction()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String("f()"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  f;\n"
  //   "}\n"
  // );

  // xml = _algebraParser.parseString(ranally::String("f(1, \"2\", three, four())"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "digraph G {\n"
  //   "  1 -> f;\n"
  //   "  \"2\" -> f;\n"
  //   "  three -> f;\n"
  //   "  four -> f;\n"
  //   "}\n"
  // );
}



void DotVisitorTest::testVisitOperator()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String("-a"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-(a)\n");

  // xml = _algebraParser.parseString(ranally::String("a + b"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "(a) + (b)\n");

  // xml = _algebraParser.parseString(ranally::String("-(a + b)"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "-((a) + (b))\n");

  // xml = _algebraParser.parseString(ranally::String("a + b * c + d"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //   "((a) + ((b) * (c))) + (d)\n");
}



void DotVisitorTest::testVisitMultipleStatements()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String("a\nb"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) == "a\nb\n");
}



void DotVisitorTest::testVisitIf()
{
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String(
  //     "if a:\n"
  //     "  b\n"
  //     "  c"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //     "if a:\n"
  //     "  b\n"
  //     "  c\n");

  // xml = _algebraParser.parseString(ranally::String(
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

  // xml = _algebraParser.parseString(ranally::String(
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
  // ranally::String xml;

  // xml = _algebraParser.parseString(ranally::String(
  //     "while a:\n"
  //     "  b\n"
  //     "  c"));
  // BOOST_CHECK(_xmlParser.parse(xml)->Accept(_visitor) ==
  //     "while a:\n"
  //     "  b\n"
  //     "  c\n");

  // xml = _algebraParser.parseString(ranally::String(
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
