#include "Ranally/Language/NumberVertexTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* NumberVertexTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<NumberVertexTest> instance(
        new NumberVertexTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &NumberVertexTest::test, instance));

    return suite;
}


NumberVertexTest::NumberVertexTest()
{
}


void NumberVertexTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
