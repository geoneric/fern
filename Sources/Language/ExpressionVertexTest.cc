#include "ExpressionVertexTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* ExpressionVertexTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<ExpressionVertexTest> instance(
    new ExpressionVertexTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &ExpressionVertexTest::test, instance));

  return suite;
}



ExpressionVertexTest::ExpressionVertexTest()
{
}



void ExpressionVertexTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

