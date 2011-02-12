#include "Ranally/Language/PurifyVisitorTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* PurifyVisitorTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<PurifyVisitorTest> instance(
    new PurifyVisitorTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &PurifyVisitorTest::test, instance));

  return suite;
}



PurifyVisitorTest::PurifyVisitorTest()
{
}



void PurifyVisitorTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

