#define BOOST_TEST_MODULE fern algorithm core copy
#include <boost/test/unit_test.hpp>
#include "fern/core/point_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/algorithm/core/copy.h"


namespace fa = fern::algorithm;


BOOST_FIXTURE_TEST_SUITE(copy, fern::ThreadClient)

void test_array_1d(
    fa::ExecutionPolicy const& execution_policy)
{
    size_t const nr_threads{fern::ThreadClient::hardware_concurrency()};
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
}


BOOST_AUTO_TEST_CASE(array_1d_parallel)
{
    test_array_1d(fa::parallel);
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

BOOST_AUTO_TEST_SUITE_END()
