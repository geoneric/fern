#define BOOST_TEST_MODULE fern fern_python_extension_algorithm_numpy_python
#include <cstdlib>
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(python)

BOOST_AUTO_TEST_CASE(unit_test)
{
    BOOST_CHECK_EQUAL(1, 1);
    std::string command{"python -m unittest discover --pattern *_test.py"};
    int result{std::system(command.c_str())};
    BOOST_CHECK_EQUAL(result, 0);
}

BOOST_AUTO_TEST_SUITE_END()
