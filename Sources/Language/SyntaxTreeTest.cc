#include "SyntaxTreeTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* SyntaxTreeTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<SyntaxTreeTest> instance(
    new SyntaxTreeTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &SyntaxTreeTest::test, instance));

  return suite;
}



SyntaxTreeTest::SyntaxTreeTest()
{
}



void SyntaxTreeTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

