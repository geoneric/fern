#include <boost/test/included/unit_test.hpp>
#include "Ranally/Interpreter/InterpreterTest.h"


boost::unit_test::test_suite* init_unit_test_suite(
    int argc,
    char** const argv) {

    struct TestSuite: public boost::unit_test::test_suite
    {
        TestSuite(
            int& /* argc */,
            char** /* argv */)
            : boost::unit_test::test_suite("Master test suite")
        {
        }
    };

    TestSuite* test = new TestSuite(argc, argv);

    test->add(InterpreterTest::suite());

    return test;
}
