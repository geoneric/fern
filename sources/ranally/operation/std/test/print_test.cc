#define BOOST_TEST_MODULE ranally operation_std
#include <boost/range/iterator_range.hpp>
#include <boost/test/unit_test.hpp>
#include "ranally/raster.h"
#include "ranally/operation/std/print.h"


BOOST_AUTO_TEST_SUITE(print)

BOOST_AUTO_TEST_CASE(print)
{
    {
        int scalar = 5;
        std::stringstream stream;
        ranally::print(scalar, stream);
        BOOST_CHECK_EQUAL(stream.str(), "5\n");
    }

    {
        int array[10] = { 1, 2, 3 };
        std::stringstream stream;
        ranally::print(
            boost::iterator_range<int*>(&array[0], array + 3), stream);
        BOOST_CHECK_EQUAL(stream.str(), "[1, 2, 3]\n");
    }

    {
        int array[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        std::stringstream stream;
        ranally::print(
            boost::iterator_range<int*>(&array[0], array + 10), stream);
        BOOST_CHECK_EQUAL(stream.str(), "[1, 2, 3, ..., 8, 9, 10]\n");
    }

    {
        ranally::Raster<int, 20, 30> raster;
        for(size_t r = 0; r < raster.nr_rows(); ++r) {
            for(size_t c = 0; c < raster.nr_cols(); ++c) {
                raster.set(r, c, r * raster.nr_cols() + c);
            }
        }
        std::stringstream stream;

        ranally::print(raster, stream);
        BOOST_WARN_EQUAL(stream.str(),
            "[[ 1,  2,  3, ..., 28, 29,  30]\n"
            " [31, 32, 33, ..., 58, 59,  60]\n"
            " [61, 62, 63, ..., 88, 89,  90]\n"
            " ...\n"
            " [61, 62, 63, ..., 88, 89, 600]\n"
        );
    }
}

BOOST_AUTO_TEST_SUITE_END()
