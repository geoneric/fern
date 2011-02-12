#include "DefinitionTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* DefinitionTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<DefinitionTest> instance(
    new DefinitionTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &DefinitionTest::test, instance));

  return suite;
}



DefinitionTest::DefinitionTest()
{
}



void DefinitionTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

