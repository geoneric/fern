#define BOOST_TEST_MODULE fern algorithm algebra vector lapacian
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array.h"
#include "fern/feature/core/array_traits.h"
#include "fern/algorithm/algebra/vector/laplacian.h"


BOOST_AUTO_TEST_SUITE(lapacian)

BOOST_AUTO_TEST_CASE(domain)
{
    {
        fern::laplacian::OutOfDomainPolicy<float> policy;
        BOOST_CHECK(policy.within_domain(5.0));
        BOOST_CHECK(policy.within_domain(0.0));
    }
}


BOOST_AUTO_TEST_CASE(range)
{
    {
        fern::laplacian::OutOfRangePolicy policy;
        BOOST_CHECK(policy.within_range(4.5));
    }
}


BOOST_AUTO_TEST_CASE(algorithm)
{

}


BOOST_AUTO_TEST_CASE(algebra)
{
    // Create input array:
    // +----+----+----+----+
    // |  0 |  1 |  2 |  3 |
    // +----+----+----+----+
    // |  4 |  5 |  6 |  7 |
    // +----+----+----+----+
    // |  8 |  9 | 10 | 11 |
    // +----+----+----+----+
    // | 12 | 13 | 14 | 15 |
    // +----+----+----+----+
    // | 16 | 17 | 18 | 19 |
    // +----+----+----+----+
    size_t const nr_rows = 5;
    size_t const nr_cols = 4;
    auto extents = fern::extents[nr_rows][nr_cols];
    fern::Array<double, 2> argument(extents);
    std::iota(argument.data(), argument.data() + argument.num_elements(), 0);

    // Calculate laplacian.
    fern::Array<double, 2> result(extents);
    fern::algebra::laplacian(argument, result);

    // Verify the result.


    // BOOST_CHECK_EQUAL(fern::get(result, 0, 0), 5);




}

BOOST_AUTO_TEST_SUITE_END()
