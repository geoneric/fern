#ifndef INCLUDED_BOOST_TEST_INCLUDED_UNIT_TEST
#include <boost/test/included/unit_test.hpp>
#define INCLUDED_BOOST_TEST_INCLUDED_UNIT_TEST
#endif

#ifndef INCLUDED_PLUSTEST
#include "PlusTest.h"
#define INCLUDED_PLUSTEST
#endif


boost::unit_test::test_suite* init_unit_test_suite(
         int argc,
         char* argv[])
{
  struct TestSuite: public boost::unit_test::test_suite
  {
    TestSuite(
         int argc,
         char** argv)
      : boost::unit_test::test_suite("Master test suite")
    {
    }
  };

  TestSuite* test = new TestSuite(argc, argv);

  test->add(PlusTest().suite());

  return test;
}

