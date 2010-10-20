#include "IfVertexTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* IfVertexTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<IfVertexTest> instance(
    new IfVertexTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &IfVertexTest::test, instance));

  return suite;
}



IfVertexTest::IfVertexTest()
{
}



void IfVertexTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

