#ifndef INCLUDED_PLUSTEST
#include "PlusTest.h"
#define INCLUDED_PLUSTEST
#endif

#ifndef INCLUDED_LIMITS
#include <limits>
#define INCLUDED_LIMITS
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

#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_PLUS
#include "Ranally/Operations/Local/Binary/Plus.h"
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_PLUS
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
  using namespace ranally::operations::binary;

  // int
  {
    plus::DomainPolicy<int> domainPolicy;

    BOOST_CHECK(domainPolicy.inDomain( 0));
    BOOST_CHECK(domainPolicy.inDomain(-1));
    BOOST_CHECK(domainPolicy.inDomain( 1));
    BOOST_CHECK(domainPolicy.inDomain(std::numeric_limits<int>::min()));
    BOOST_CHECK(domainPolicy.inDomain(std::numeric_limits<int>::max()));
  }

  // unsigned int
  {
    plus::DomainPolicy<unsigned int> domainPolicy;

    BOOST_CHECK(domainPolicy.inDomain(0u));
    BOOST_CHECK(domainPolicy.inDomain(1u));
    BOOST_CHECK(domainPolicy.inDomain(
         std::numeric_limits<unsigned int>::min()));
    BOOST_CHECK(domainPolicy.inDomain(
         std::numeric_limits<unsigned int>::max()));
  }

  // float
  {
    plus::DomainPolicy<double> domainPolicy;

    BOOST_CHECK(domainPolicy.inDomain( 0.0));
    BOOST_CHECK(domainPolicy.inDomain(-1.0));
    BOOST_CHECK(domainPolicy.inDomain( 1.0));
    BOOST_CHECK(domainPolicy.inDomain( std::numeric_limits<double>::min()));
    BOOST_CHECK(domainPolicy.inDomain(-std::numeric_limits<double>::min()));
    BOOST_CHECK(domainPolicy.inDomain( std::numeric_limits<double>::max()));
    BOOST_CHECK(domainPolicy.inDomain(-std::numeric_limits<double>::max()));
    // TODO BOOST_CHECK(domainPolicy.inDomain(nan);
  }
}



void PlusTest::testAlgorithm()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}



void PlusTest::testRange()
{
  using namespace ranally::operations::binary;

  // int
  {
    plus::RangePolicy<int> rangePolicy;

    BOOST_CHECK( rangePolicy.inRange( 0,  0,  0));
    BOOST_CHECK( rangePolicy.inRange( 3,  4,  7));
    BOOST_CHECK( rangePolicy.inRange( 3, -4, -1));
    BOOST_CHECK( rangePolicy.inRange(-3, -4, -7));

    BOOST_CHECK(!rangePolicy.inRange( 3,  4, -7));
    BOOST_CHECK(!rangePolicy.inRange(-3, -4,  7));
  }

  // unsigned int
  {
    // TODO
  }

  // float
  {
    // TODO
  }
}

