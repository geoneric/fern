#include "NameVertexTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* NameVertexTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<NameVertexTest> instance(
    new NameVertexTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &NameVertexTest::test, instance));

  return suite;
}



NameVertexTest::NameVertexTest()
{
}



void NameVertexTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

