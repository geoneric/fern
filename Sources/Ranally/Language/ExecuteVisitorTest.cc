#include "Ranally/Language/ExecuteVisitorTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* ExecuteVisitorTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<ExecuteVisitorTest> instance(
        new ExecuteVisitorTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &ExecuteVisitorTest::test, instance));

    return suite;
}


ExecuteVisitorTest::ExecuteVisitorTest()
{
}


void ExecuteVisitorTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
