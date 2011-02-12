#include "Ranally/Language/StringVertexTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* StringVertexTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<StringVertexTest> instance(
    new StringVertexTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &StringVertexTest::test, instance));

  return suite;
}



StringVertexTest::StringVertexTest()
{
}



void StringVertexTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

