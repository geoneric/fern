#define BOOST_TEST_MODULE fern algorithm algebra vector lapacian
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/core/type_traits.h"
#include "fern/core/types.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/algebra/vector/laplacian.h"


BOOST_AUTO_TEST_SUITE(lapacian)

BOOST_AUTO_TEST_CASE(traits)
{
    using Laplacian = fern::algebra::Laplacian<fern::f32, fern::f32>;
    BOOST_CHECK((std::is_same<fern::OperationTraits<Laplacian>::category,
        fern::neighborhood_operation_tag>::value));
}


template<
    class Value>
using OutOfDomainPolicy = fern::laplacian::OutOfDomainPolicy<Value>;


BOOST_AUTO_TEST_CASE(out_of_domain_policy)
{
    {
        OutOfDomainPolicy<fern::float32_t> policy;
        BOOST_CHECK(policy.within_domain(5.0));
        BOOST_CHECK(policy.within_domain(-5.0));
        BOOST_CHECK(policy.within_domain(0.0));
    }
}


template<
    class Result>
using OutOfRangePolicy = fern::laplacian::OutOfRangePolicy<Result>;


BOOST_AUTO_TEST_CASE(out_of_range_policy)
{
    {
        OutOfRangePolicy<fern::float32_t> policy;
        BOOST_CHECK(policy.within_range(4.5));
        BOOST_CHECK(!policy.within_range(fern::nan<fern::float32_t>()));
        BOOST_CHECK(!policy.within_range(fern::infinity<fern::float32_t>()));
    }
}


template<
    class Values,
    class Results>
void verify_value(
    Values const& values,
    Results const& results_we_want)
{
    Results results_we_get;
    fern::algebra::laplacian(values, results_we_get);
    // BOOST_CHECK_EQUAL(results_we_get, results_we_want);
}


template<
    class T>
using MaskedRaster = fern::MaskedRaster<T, 2>;


BOOST_AUTO_TEST_CASE(algorithm)
{
    // Create input raster:
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

    double const cell_size = 1.0;
    double const cell_width = cell_size;
    double const cell_height = cell_size;
    double const west = 0.0;
    double const north = 0.0;

    MaskedRaster<double>::Transformation transformation{{west, cell_width,
        north, cell_height}};
    MaskedRaster<double> raster(extents, transformation);

    std::iota(raster.data(), raster.data() + raster.num_elements(), 0);

    // Calculate laplacian.
    MaskedRaster<double> result(extents, transformation);
    fern::algebra::laplacian(raster, result);


    // Verify the result.





    // BOOST_CHECK_EQUAL(fern::get(result, 0, 0), 5);

    // fern::algebra::laplacian(values, results_we_get);
}


/// BOOST_AUTO_TEST_CASE(algebra)
/// {
///     // Create input array:
///     // +----+----+----+----+
///     // |  0 |  1 |  2 |  3 |
///     // +----+----+----+----+
///     // |  4 |  5 |  6 |  7 |
///     // +----+----+----+----+
///     // |  8 |  9 | 10 | 11 |
///     // +----+----+----+----+
///     // | 12 | 13 | 14 | 15 |
///     // +----+----+----+----+
///     // | 16 | 17 | 18 | 19 |
///     // +----+----+----+----+
///     size_t const nr_rows = 5;
///     size_t const nr_cols = 4;
///     auto extents = fern::extents[nr_rows][nr_cols];
///     fern::Array<double, 2> argument(extents);
///     std::iota(argument.data(), argument.data() + argument.num_elements(), 0);
/// 
///     // Calculate laplacian.
///     fern::Array<double, 2> result(extents);
///     fern::algebra::laplacian(argument, result);
/// 
///     // Verify the result.
/// 
/// 
///     // BOOST_CHECK_EQUAL(fern::get(result, 0, 0), 5);
/// 
/// 
/// 
/// 
/// }

BOOST_AUTO_TEST_SUITE_END()
