#include "Ranally/Language/StatementVertexTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* StatementVertexTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<StatementVertexTest> instance(
    new StatementVertexTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &StatementVertexTest::test, instance));

  return suite;
}



StatementVertexTest::StatementVertexTest()
{
}



void StatementVertexTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

