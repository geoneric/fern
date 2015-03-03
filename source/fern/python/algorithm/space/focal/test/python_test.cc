#define BOOST_TEST_MODULE fern fern_python_extension_algorithm_space_focal
#include <cstdlib>
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(python)

BOOST_AUTO_TEST_CASE(unit_test)
{
    std::string command{"python -m unittest discover --pattern '*_test.py'"};
    int result{std::system(command.c_str())};
    BOOST_CHECK_EQUAL(result, 0);
}

BOOST_AUTO_TEST_SUITE_END()
