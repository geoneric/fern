#include "Ranally/Interpreter/InterpreterTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* InterpreterTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<InterpreterTest> instance(
        new InterpreterTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &InterpreterTest::test, instance));

    return suite;
}


InterpreterTest::InterpreterTest()
{
}


void InterpreterTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
