#include "AverageTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Algorithm/Average.h"


boost::unit_test::test_suite* AverageTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<AverageTest> instance(
        new AverageTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &AverageTest::test, instance));

    return suite;
}


AverageTest::AverageTest()
{
}


void AverageTest::test()
{
    // int + int -> int
    {
    }
}
