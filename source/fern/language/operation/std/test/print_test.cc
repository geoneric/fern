// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern operation std print
#include <boost/range/iterator_range.hpp>
#include <boost/test/unit_test.hpp>
#include "fern/language/operation/raster.h"
#include "fern/language/operation/std/print.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_CASE(print)
{
    {
        int constant = 5;
        std::stringstream stream;
        fl::print(constant, stream);
        BOOST_CHECK_EQUAL(stream.str(), "5\n");
    }

    {
        int array[10] = { 1, 2, 3 };
        std::stringstream stream;
        fl::print(
            boost::iterator_range<int*>(&array[0], array + 3), stream);
        BOOST_CHECK_EQUAL(stream.str(), "[1, 2, 3]\n");
    }

    {
        int array[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        std::stringstream stream;
        fl::print(
            boost::iterator_range<int*>(&array[0], array + 10), stream);
        BOOST_CHECK_EQUAL(stream.str(), "[1, 2, 3, ..., 8, 9, 10]\n");
    }

    {
        fl::Raster<int, 20, 30> raster;
        for(size_t r = 0; r < raster.nr_rows(); ++r) {
            for(size_t c = 0; c < raster.nr_cols(); ++c) {
                raster.set(r, c, r * raster.nr_cols() + c);
            }
        }
        std::stringstream stream;

        fl::print(raster, stream);
        BOOST_WARN_EQUAL(stream.str(),
            "[[ 1,  2,  3, ..., 28, 29,  30]\n"
            " [31, 32, 33, ..., 58, 59,  60]\n"
            " [61, 62, 63, ..., 88, 89,  90]\n"
            " ...\n"
            " [61, 62, 63, ..., 88, 89, 600]\n"
        );
    }
}
