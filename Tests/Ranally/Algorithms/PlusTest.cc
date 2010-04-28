#ifndef INCLUDED_PLUSTEST
#include "PlusTest.h"
#define INCLUDED_PLUSTEST
#endif

#ifndef INCLUDED_BOOST_SHARED_PTR
#include <boost/shared_ptr.hpp>
#define INCLUDED_BOOST_SHARED_PTR
#endif

#ifndef INCLUDED_BOOST_TEST_TEST_TOOLS
#include <boost/test/test_tools.hpp>
#define INCLUDED_BOOST_TEST_TEST_TOOLS
#endif

#ifndef INCLUDED_BOOST_TEST_UNIT_TEST_SUITE
#include <boost/test/unit_test_suite.hpp>
#define INCLUDED_BOOST_TEST_UNIT_TEST_SUITE
#endif



//------------------------------------------------------------------------------
// DEFINITION OF STATIC PLUSTEST MEMBERS
//------------------------------------------------------------------------------

//! Suite.
boost::unit_test::test_suite* PlusTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<PlusTest> instance(
         new PlusTest());
  suite->add(BOOST_CLASS_TEST_CASE(
         &PlusTest::testDomain, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &PlusTest::testAlgorithm, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &PlusTest::testRange, instance));

  return suite;
}



//------------------------------------------------------------------------------
// DEFINITION OF PLUSTEST MEMBERS
//------------------------------------------------------------------------------

//! Constructor.
PlusTest::PlusTest()
{
}



void PlusTest::testDomain()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}



void PlusTest::testAlgorithm()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}



void PlusTest::testRange()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

