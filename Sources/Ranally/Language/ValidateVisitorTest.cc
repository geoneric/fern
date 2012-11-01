#include "Ranally/Language/ValidateVisitorTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* ValidateVisitorTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<ValidateVisitorTest> instance(
        new ValidateVisitorTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &ValidateVisitorTest::test, instance));

    return suite;
}


ValidateVisitorTest::ValidateVisitorTest()
{
}


void ValidateVisitorTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
