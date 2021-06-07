// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <boost/test/unit_test.hpp>
#include <iomanip>
#include <iostream>
#include "fern/feature/core/masked_scalar.h"
#include "fern/feature/core/masked_raster.h"
#include "fern/algorithm/algebra/elementary/equal.h"
#include "fern/algorithm/statistic/count.h"


// Macro to create test cases based on a name passed in. This name must
// correspond with a template function named test_<name>. The one template
// argument must be the execution policy's type, which is passed in.
// The template function will be called 4 times, with different execution
// policies:
// - fern::algorithm::SequentialExecutionPolicy
// - fern::algorithm::ExecutionPolicy{fern::algorithm::SequentialExecutionPolicy}
// - fern::algorithm::ParallelExecutionPolicy
// - fern::algorithm::ExecutionPolicy{fern::algorithm::ParallelExecutionPolicy}
#define FERN_TEST_CASES(name)                              \
BOOST_AUTO_TEST_CASE(array_##name##_sequential)            \
{                                                          \
    fern::algorithm::SequentialExecutionPolicy sequential; \
    test_##name(sequential);                               \
    fern::algorithm::ExecutionPolicy execution_policy{     \
        sequential};                                       \
    test_##name(execution_policy);                         \
}                                                          \
                                                           \
BOOST_AUTO_TEST_CASE(array_##name##_parallel)              \
{                                                          \
    fern::algorithm::ParallelExecutionPolicy parallel;     \
    test_##name(parallel);                                 \
    fern::algorithm::ExecutionPolicy execution_policy{     \
        parallel};                                         \
    test_##name(execution_policy);                         \
}


#define FERN_UNARY_AGGREGATE_TEST_CASES()  \
FERN_TEST_CASES(0d_0d)                     \
FERN_TEST_CASES(0d_0d_masked)              \
FERN_TEST_CASES(1d_0d)                     \
FERN_TEST_CASES(1d_0d_masked)              \
FERN_TEST_CASES(2d_0d)                     \
FERN_TEST_CASES(2d_0d_masked)              \


#define FERN_BINARY_TEST_CASES()     \
FERN_TEST_CASES(0d_0d_0d)            \
FERN_TEST_CASES(1d_1d_1d)            \
FERN_TEST_CASES(1d_1d_1d_masked)     \
FERN_TEST_CASES(1d_1d_0d)            \
FERN_TEST_CASES(1d_1d_0d_masked)     \
FERN_TEST_CASES(1d_0d_1d)            \
FERN_TEST_CASES(1d_0d_1d_masked)     \
FERN_TEST_CASES(2d_2d_2d)            \
FERN_TEST_CASES(2d_2d_2d_masked)     \
FERN_TEST_CASES(2d_2d_2d)            \
FERN_TEST_CASES(2d_2d_0d)            \
FERN_TEST_CASES(2d_2d_0d_masked)     \
FERN_TEST_CASES(2d_0d_2d)            \
FERN_TEST_CASES(2d_0d_2d_masked)


namespace fern {
namespace test {

static size_t const nr_cores{fern::hardware_concurrency()};
// Using nr_cores to scale the test data size messes things up on machines
// with many cores. Tests will fail because suddenly results end up
// being out of range. Use a fixed number, like 8.
static size_t const nr_rows{30 * 8};  // nr_cores};
static size_t const nr_cols{20 * 8};  // nr_cores};
static size_t const nr_elements_1d{10 * 8};  // nr_cores};
static size_t const nr_elements_2d{nr_rows * nr_cols};

} // namespace test


template<
    class ExecutionPolicy,
    class Value>
bool compare(
    ExecutionPolicy& execution_policy,
    Value const& value1,
    Value const& value2)
{
    auto equal_result = clone<int>(value1, 0);

    algorithm::algebra::equal(execution_policy, value1, value2, equal_result);

    uint64_t nr_equal_values;

    algorithm::statistic::count(execution_policy, equal_result, 1,
        nr_equal_values);

    BOOST_CHECK_EQUAL(nr_equal_values, size(value1));

    return nr_equal_values == size(value1);
}


template<
    class ExecutionPolicy,
    class Value>
bool compare(
    ExecutionPolicy& execution_policy,
    MaskedScalar<Value> const& value1,
    MaskedScalar<Value> const& value2)
{
    bool values_are_the_same = compare(execution_policy,
            value1.value(), value2.value());
    bool mask_is_the_same = compare(execution_policy,
            value1.mask(), value2.mask());

    BOOST_CHECK(values_are_the_same);
    BOOST_CHECK(mask_is_the_same);

    return values_are_the_same && mask_is_the_same;
}


template<
    class ExecutionPolicy,
    class Value,
    size_t nr_dimensions>
bool compare(
    ExecutionPolicy& execution_policy,
    MaskedArray<Value, nr_dimensions> const& value1,
    MaskedArray<Value, nr_dimensions> const& value2)
{
    bool values_are_the_same = compare(execution_policy,
            dynamic_cast<Array<Value, nr_dimensions> const&>(value1),
            dynamic_cast<Array<Value, nr_dimensions> const&>(value2));
    bool mask_is_the_same = compare(execution_policy,
            value1.mask(),
            value2.mask());

    BOOST_CHECK(values_are_the_same);
    BOOST_CHECK(mask_is_the_same);

    return values_are_the_same && mask_is_the_same;
}


template<
    class ExecutionPolicy,
    class Value,
    size_t nr_dimensions>
bool compare(
    ExecutionPolicy& execution_policy,
    MaskedRaster<Value, nr_dimensions> const& value1,
    MaskedRaster<Value, nr_dimensions> const& value2)
{
    bool values_are_the_same = compare(execution_policy,
            dynamic_cast<Array<Value, nr_dimensions> const&>(value1),
            dynamic_cast<Array<Value, nr_dimensions> const&>(value2));
    bool mask_is_the_same = compare(execution_policy,
            value1.mask(),
            value2.mask());

    BOOST_CHECK(values_are_the_same);
    BOOST_CHECK(mask_is_the_same);

    return values_are_the_same && mask_is_the_same;
}


template<
    class T>
inline std::ostream& operator<<(
    std::ostream& stream,
    MaskedScalar<T> const& constant)
{
    stream << constant.value() << (constant.mask() ? "(masked)" : "");
    return stream;
}


template<
    class T,
    size_t nr_dimensions>
inline std::ostream& operator<<(
    std::ostream& stream,
    MaskedArray<T, nr_dimensions> const& array)
{
    for(size_t r = 0; r < size(array, 0); ++r) {
        for(size_t c = 0; c < size(array, 1); ++c) {
            stream
                << (array.mask()[r][c] ? "(" : " ")
                << std::setw(5) << array[r][c]
                << (array.mask()[r][c] ? ")" : " ")
                << " ";
        }
        stream << "\n";
    }

    return stream;
}


template<
    class T,
    size_t nr_dimensions>
inline std::ostream& operator<<(
    std::ostream& stream,
    MaskedRaster<T, nr_dimensions> const& raster)
{
    for(size_t r = 0; r < size(raster, 0); ++r) {
        for(size_t c = 0; c < size(raster, 1); ++c) {
            stream
                << (raster.mask()[r][c] ? "(" : " ")
                << std::setw(5) << raster[r][c]
                << (raster.mask()[r][c] ? ")" : " ")
                << " ";
        }
        stream << "\n";
    }

    return stream;
}


inline std::ostream& operator<<(
    std::ostream& stream,
    uint8_t const& value)
{
    stream << static_cast<int64_t>(value);
    return stream;
}

}


namespace boost {
namespace test_tools {
namespace tt_detail {

// http://stackoverflow.com/questions/17572583/boost-check-fails-to-compile-operator-for-custom-types
template<>
inline std::ostream& operator<<(
    std::ostream& stream,
    print_helper_t<uint8_t> const& print_helper)
{
    stream << static_cast<int16_t>(print_helper.m_t);
    return stream;
}

} // namespace tt_detail
} // namespace test_tools
} // namespace boost
