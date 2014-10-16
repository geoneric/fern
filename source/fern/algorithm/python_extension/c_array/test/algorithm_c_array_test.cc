#define BOOST_TEST_MODULE fern algorithm c_array
#include <memory>
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/python_extension/c_array/algorithm.h"


BOOST_FIXTURE_TEST_SUITE(c_array, fern::ThreadClient)

BOOST_AUTO_TEST_CASE(add)
{
    {
        size_t const size{100};

        std::unique_ptr<double> value1{new double[size]};
        std::iota(value1.get(), value1.get() + size, 0.0);

        double const value2{5.0};

        std::unique_ptr<double> result{new double[size]};

        auto value1_reference(fern::ArrayReference<double, 1>(
            value1.get(), fern::extents[size]));
        auto result_reference(fern::ArrayReference<double, 1>(
            result.get(), fern::extents[size]));

        fern::add(result_reference, value1_reference, value2);

        // TODO Add tests.
        std::cout << result.get()[0] << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()
