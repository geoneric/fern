// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern fern_python_extension_algorithm_core
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
