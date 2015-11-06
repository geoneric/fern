// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core copy
#include <boost/test/unit_test.hpp>
#include <numeric>
#include "fern/core/data_type_traits/point.h"
#include "fern/core/data_customization_point/vector.h"
#include "fern/algorithm/core/copy.h"


namespace fa = fern::algorithm;


template<
    typename ExecutionPolicy>
void test_array_1d(
    ExecutionPolicy& execution_policy)
{
    size_t const nr_threads{fern::hardware_concurrency()};
    size_t const nr_elements{10 * nr_threads};
    std::vector<int> source(nr_elements);
    std::vector<int> result_we_want(nr_elements);
    std::vector<int> result_we_got(nr_elements);

    // 0, 1, 2, 3, ..., n-1
    std::iota(source.begin(), source.end(), 0);

    {
        size_t const nr_elements_to_copy{2 * nr_threads};
        fa::IndexRanges<1> range{
            fa::IndexRange(3, 3 + nr_elements_to_copy)
        };
        fern::Point<size_t, 1> position{nr_elements_to_copy};

        // 0, ..., 0, 0, 1, 2, 3, ..., 0
        std::fill(result_we_want.begin(), result_we_want.end(), 0);
        std::copy(source.begin() + 3, source.begin() + 3 + nr_elements_to_copy,
            result_we_want.begin() + position[0]);

        fa::core::copy(execution_policy, source, range, result_we_got,
            position);

        BOOST_CHECK(result_we_got == result_we_want);
    }
}


BOOST_AUTO_TEST_CASE(array_1d_sequential)
{
    test_array_1d(fa::sequential);
    fa::ExecutionPolicy execution_policy{fa::sequential};
    test_array_1d(execution_policy);
}


BOOST_AUTO_TEST_CASE(array_1d_parallel)
{
    test_array_1d(fa::parallel);
    fa::ExecutionPolicy execution_policy{fa::parallel};
    test_array_1d(execution_policy);
}


void test_array_1d_masked(
    fa::ExecutionPolicy const& /* execution_policy */)
{
    // TODO
}


BOOST_AUTO_TEST_CASE(array_1d_masked_sequential)
{
    test_array_1d_masked(fa::sequential);
}


BOOST_AUTO_TEST_CASE(array_1d_masked_parallel)
{
    test_array_1d_masked(fa::parallel);
}
