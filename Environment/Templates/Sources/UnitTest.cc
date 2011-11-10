#include "pack_ClassTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* ClassTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<ClassTest> instance(
    new ClassTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &ClassTest::test, instance));

  return suite;
}



ClassTest::ClassTest()
{
}



void ClassTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

