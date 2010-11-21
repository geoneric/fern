#include "SymbolTableTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* SymbolTableTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<SymbolTableTest> instance(
    new SymbolTableTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &SymbolTableTest::test, instance));

  return suite;
}



SymbolTableTest::SymbolTableTest()
{
}



void SymbolTableTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

