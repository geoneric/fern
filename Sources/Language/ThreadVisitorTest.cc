#include "ThreadVisitorTest.h"

#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include "dev_UnicodeUtils.h"

#include "ScriptVertex.h"



boost::unit_test::test_suite* ThreadVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<ThreadVisitorTest> instance(
    new ThreadVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitEmptyScript, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitName, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitAssignment, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitString, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitNumber, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitFunction, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitOperator, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitMultipleStatements, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitIf, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &ThreadVisitorTest::testVisitWhile, instance));

  return suite;
}



ThreadVisitorTest::ThreadVisitorTest()
{
}



void ThreadVisitorTest::testVisitEmptyScript()
{
  boost::shared_ptr<ranally::ScriptVertex> tree;

  tree = _xmlParser.parse(_algebraParser.parseString(UnicodeString("")));
  tree->Accept(_visitor);
  BOOST_CHECK_EQUAL(tree->successor(), &(*tree));
}



void ThreadVisitorTest::testVisitName()
{
}



void ThreadVisitorTest::testVisitAssignment()
{
}



void ThreadVisitorTest::testVisitString()
{
}



void ThreadVisitorTest::testVisitNumber()
{
}



void ThreadVisitorTest::testVisitFunction()
{
}



void ThreadVisitorTest::testVisitOperator()
{
}



void ThreadVisitorTest::testVisitMultipleStatements()
{
}



void ThreadVisitorTest::testVisitIf()
{
}



void ThreadVisitorTest::testVisitWhile()
{
}
