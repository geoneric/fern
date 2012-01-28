#include "Ranally/Language/OptimizeVisitorTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* OptimizeVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<OptimizeVisitorTest> instance(
    new OptimizeVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &OptimizeVisitorTest::test, instance));

  return suite;
}



OptimizeVisitorTest::OptimizeVisitorTest()
{
}



void OptimizeVisitorTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

