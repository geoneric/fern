#define BOOST_TEST_MODULE fern algorithm algebra equal
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_traits.h"
/// #include "fern/feature/core/array_view_traits.h"
/// #include "fern/feature/core/masked_array_traits.h"
#include "fern/core/vector_traits.h"
/// #include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/equal.h"
#include "fern/core/typename.h"


#define verify_result_value_type(                                              \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    typedef fern::algebra::Equal<A1, A2>::R TypeWeGet;                         \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        fern::typename_<TypeWeGet>() + " != " +                                \
        fern::typename_<TypeWeWant>());                                        \
}


template<
    class A1,
    class A2,
    class R>
void verify_value(
    A1 const& argument1,
    A2 const& argument2,
    R const& result_we_want)
{
    fern::algebra::Equal<A1, A2> operation;
    R result_we_get;

    operation(argument1, argument2, result_we_get);
    BOOST_CHECK_EQUAL(result_we_get, result_we_want);
}


namespace d1 {

template<
    class T>
using Array = fern::Array<T, 1>;

} // namespace d1


namespace d2 {

template<
    class T>
using Array = fern::Array<T, 2>;

} // namespace d2


BOOST_AUTO_TEST_SUITE(equal)

BOOST_AUTO_TEST_CASE(result_type)
{
    verify_result_value_type(uint8_t, uint8_t, bool);
    verify_result_value_type(bool, bool, bool);
    verify_result_value_type(double, int32_t, bool);

    verify_result_value_type(std::vector<uint8_t>, std::vector<uint8_t>,
        std::vector<bool>);
    verify_result_value_type(d1::Array<uint8_t>, d1::Array<uint8_t>,
        d1::Array<bool>);
    verify_result_value_type(d2::Array<uint8_t>, d2::Array<uint8_t>,
        d2::Array<bool>);
}


BOOST_AUTO_TEST_CASE(constants)
{
    verify_value<int8_t, int8_t>(-5, 6, false);
    verify_value<int8_t, int8_t>(-5, -5, true);
    verify_value<double, double>(-5.5, -5.5, true);
    verify_value<double, double>(-5.5, -5.4, false);
}


BOOST_AUTO_TEST_CASE(collections)
{
    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    auto extents = fern::extents[nr_rows][nr_cols];

    fern::Array<int8_t, 2> argument1(extents);
    argument1[0][0] = -2;
    argument1[0][1] = -1;
    argument1[1][0] =  0;
    argument1[1][1] =  9;
    argument1[2][0] =  1;
    argument1[2][1] =  2;

    fern::Array<int8_t, 2> argument2(extents);
    argument2[0][0] = -1;
    argument2[0][1] = -1;
    argument2[1][0] =  0;
    argument2[1][1] =  9;
    argument1[2][0] =  2;
    argument2[2][1] =  2;

    int8_t argument3 = 2;

    // masked_array == masked_array
    {
        fern::Array<bool, 2> result(extents);
        fern::algebra::equal(argument1, argument2, result);

        BOOST_CHECK(!result[0][0]);
        BOOST_CHECK( result[0][1]);
        BOOST_CHECK( result[1][0]);
        BOOST_CHECK( result[1][1]);
        BOOST_CHECK(!result[2][0]);
        BOOST_CHECK( result[2][1]);
    }

    /// // masked_array + 5
    /// {
    ///     // Create room for the result.
    ///     // Set the mask.
    ///     typedef fern::result<int8_t, int8_t>::type R;
    ///     fern::MaskedArray<R, 2> result(extents);
    ///     result.set_mask(argument1.mask(), true);

    ///     typedef decltype(argument1) A1;
    ///     typedef decltype(argument3) A2;
    ///     typedef fern::ArgumentTraits<A1>::value_type A1Value;
    ///     typedef fern::ArgumentTraits<A2>::value_type A2Value;
    ///     typedef fern::DiscardDomainErrors<A1Value, A2Value> OutOfDomainPolicy;
    ///     typedef fern::plus::OutOfRangePolicy<A1Value, A2Value> OutOfRangePolicy;
    ///     typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
    ///     typedef fern::algebra::Plus<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
    ///         NoDataPolicy> Plus;

    ///     Plus plus(NoDataPolicy(result.mask(), true));

    ///     plus(argument1, argument3, result);

    ///     BOOST_CHECK(!result.mask()[0][0]);
    ///     BOOST_CHECK_EQUAL(result[0][0], 3);

    ///     BOOST_CHECK(!result.mask()[0][1]);
    ///     BOOST_CHECK_EQUAL(result[0][1], 4);

    ///     BOOST_CHECK(!result.mask()[1][0]);
    ///     BOOST_CHECK_EQUAL(result[1][0], 5);

    ///     BOOST_CHECK( result.mask()[1][1]);

    ///     BOOST_CHECK(!result.mask()[2][0]);
    ///     BOOST_CHECK_EQUAL(result[2][0], 6);

    ///     BOOST_CHECK(!result.mask()[2][1]);
    ///     BOOST_CHECK_EQUAL(result[2][1], 7);
    /// }
}


/// BOOST_AUTO_TEST_CASE(argument_types)
/// {
///     // Verify that we can pass in all kinds of collection types.
/// 
///     // constant + constant
///     {
///         uint8_t argument1(5);
///         uint8_t argument2(6);
///         typedef fern::result<uint8_t, uint8_t>::type R;
///         R result;
/// 
///         fern::algebra::plus(argument1, argument2, result);
/// 
///         BOOST_CHECK_EQUAL(result, 11u);
///     }
/// 
///     // constant + vector
///     {
///         uint8_t argument1(5);
///         std::vector<uint8_t> argument2({1, 2, 3});
///         typedef fern::result<uint8_t, uint8_t>::type R;
///         std::vector<R> result(argument2.size());
/// 
///         fern::algebra::plus(argument1, argument2, result);
/// 
///         BOOST_REQUIRE_EQUAL(result.size(), 3u);
///         BOOST_CHECK_EQUAL(result[0], 6u);
///         BOOST_CHECK_EQUAL(result[1], 7u);
///         BOOST_CHECK_EQUAL(result[2], 8u);
///     }
/// 
///     // vector + constant
///     {
///         std::vector<uint8_t> argument1({1, 2, 3});
///         uint8_t argument2(5);
///         typedef fern::result<uint8_t, uint8_t>::type R;
///         std::vector<R> result(argument1.size());
/// 
///         fern::algebra::plus(argument1, argument2, result);
/// 
///         BOOST_REQUIRE_EQUAL(result.size(), 3u);
///         BOOST_CHECK_EQUAL(result[0], 6u);
///         BOOST_CHECK_EQUAL(result[1], 7u);
///         BOOST_CHECK_EQUAL(result[2], 8u);
///     }
/// 
///     // vector + vector
///     {
///         std::vector<uint8_t> argument1({1, 2, 3});
///         std::vector<uint8_t> argument2({4, 5, 6});
///         typedef fern::result<uint8_t, uint8_t>::type R;
///         std::vector<R> result(argument1.size());
/// 
///         fern::algebra::plus(argument1, argument2, result);
/// 
///         BOOST_REQUIRE_EQUAL(result.size(), 3u);
///         BOOST_CHECK_EQUAL(result[0], 5u);
///         BOOST_CHECK_EQUAL(result[1], 7u);
///         BOOST_CHECK_EQUAL(result[2], 9u);
///     }
/// 
///     // array + array
///     {
///         fern::Array<int8_t, 2> argument(fern::extents[3][2]);
///         argument[0][0] = -2;
///         argument[0][1] = -1;
///         argument[1][0] =  0;
///         argument[1][1] =  9;
///         argument[2][0] =  1;
///         argument[2][1] =  2;
///         typedef fern::result<int8_t, int8_t>::type R;
///         fern::Array<R, 2> result(fern::extents[3][2]);
/// 
///         fern::algebra::plus(argument, argument, result);
/// 
///         BOOST_CHECK_EQUAL(result[0][0], -4);
///         BOOST_CHECK_EQUAL(result[0][1], -2);
///         BOOST_CHECK_EQUAL(result[1][0],  0);
///         BOOST_CHECK_EQUAL(result[1][1], 18);
///         BOOST_CHECK_EQUAL(result[2][0],  2);
///         BOOST_CHECK_EQUAL(result[2][1],  4);
///     }
/// 
///     // masked_array + masked_array
///     {
///         fern::MaskedArray<int8_t, 2> argument(fern::extents[3][2]);
///         argument[0][0] = -2;
///         argument[0][1] = -1;
///         argument[1][0] =  0;
///         argument.mask()[1][1] =  true;
///         argument[1][1] =  9;
///         argument[2][0] =  1;
///         argument[2][1] =  2;
///         typedef fern::result<int8_t, int8_t>::type R;
///         fern::MaskedArray<R, 2> result(fern::extents[3][2]);
/// 
///         fern::algebra::plus(argument, argument, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], -4);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], -2);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0],  0);
/// 
///         // Although the input data has a mask, the default policy discards
///         // it. So the result doesn't have masked values.
///         BOOST_CHECK(!result.mask()[1][1]);
///         BOOST_CHECK_EQUAL(result[1][1], 18);
/// 
///         BOOST_CHECK(!result.mask()[2][0]);
///         BOOST_CHECK_EQUAL(result[2][0],  2);
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1],  4);
///     }
/// }
/// 
/// 
/// BOOST_AUTO_TEST_CASE(no_data)
/// {
///     size_t const nr_rows = 3;
///     size_t const nr_cols = 2;
///     auto extents = fern::extents[nr_rows][nr_cols];
/// 
///     fern::MaskedArray<int8_t, 2> argument1(extents);
///     argument1[0][0] = -2;
///     argument1[0][1] = -1;
///     argument1[1][0] =  0;
///     argument1.mask()[1][1] =  true;
///     argument1[2][0] =  1;
///     argument1[2][1] =  2;
/// 
///     fern::MaskedArray<int8_t, 2> argument2(extents);
///     argument2[0][0] = -2;
///     argument2[0][1] = -1;
///     argument2[1][0] =  0;
///     argument2[1][1] =  9;
///     argument2.mask()[2][0] =  true;
///     argument2[2][1] =  2;
/// 
///     int8_t argument3 = 5;
/// 
///     // masked_array + masked_array
///     {
///         // Create room for the result.
///         // Set the mask.
///         typedef fern::result<int8_t, int8_t>::type R;
///         fern::MaskedArray<R, 2> result(extents);
///         result.set_mask(argument1.mask(), true);
///         result.set_mask(argument2.mask(), true);
/// 
///         typedef decltype(argument1) A1;
///         typedef decltype(argument2) A2;
///         typedef fern::ArgumentTraits<A1>::value_type A1Value;
///         typedef fern::ArgumentTraits<A2>::value_type A2Value;
///         typedef fern::DiscardDomainErrors<A1Value, A2Value> OutOfDomainPolicy;
///         typedef fern::plus::OutOfRangePolicy<A1Value, A2Value> OutOfRangePolicy;
///         typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
///         typedef fern::algebra::Plus<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
///             NoDataPolicy> Plus;
/// 
///         Plus plus(NoDataPolicy(result.mask(), true));
/// 
///         plus(argument1, argument2, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], -4);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], -2);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0],  0);
/// 
///         BOOST_CHECK( result.mask()[1][1]);
///         // Value is masked: it is undefined.
///         // BOOST_CHECK_EQUAL(result[1][1], 18);
/// 
///         BOOST_CHECK(result.mask()[2][0]);
///         // Value is masked.
///         // BOOST_CHECK_EQUAL(result[2][0],  2);
/// 
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1],  4);
///     }
/// 
///     // masked_array + 5
///     {
///         // Create room for the result.
///         // Set the mask.
///         typedef fern::result<int8_t, int8_t>::type R;
///         fern::MaskedArray<R, 2> result(extents);
///         result.set_mask(argument1.mask(), true);
/// 
///         typedef decltype(argument1) A1;
///         typedef decltype(argument3) A2;
///         typedef fern::ArgumentTraits<A1>::value_type A1Value;
///         typedef fern::ArgumentTraits<A2>::value_type A2Value;
///         typedef fern::DiscardDomainErrors<A1Value, A2Value> OutOfDomainPolicy;
///         typedef fern::plus::OutOfRangePolicy<A1Value, A2Value> OutOfRangePolicy;
///         typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
///         typedef fern::algebra::Plus<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
///             NoDataPolicy> Plus;
/// 
///         Plus plus(NoDataPolicy(result.mask(), true));
/// 
///         plus(argument1, argument3, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], 3);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], 4);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0], 5);
/// 
///         BOOST_CHECK( result.mask()[1][1]);
/// 
///         BOOST_CHECK(!result.mask()[2][0]);
///         BOOST_CHECK_EQUAL(result[2][0], 6);
/// 
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1], 7);
///     }
/// 
///     // 5 + masked_array
///     {
///         // Create room for the result.
///         // Set the mask.
///         typedef fern::result<int8_t, int8_t>::type R;
///         fern::MaskedArray<R, 2> result(extents);
///         result.set_mask(argument1.mask(), true);
/// 
///         typedef decltype(argument3) A1;
///         typedef decltype(argument1) A2;
///         typedef fern::ArgumentTraits<A1>::value_type A1Value;
///         typedef fern::ArgumentTraits<A2>::value_type A2Value;
///         typedef fern::DiscardDomainErrors<A1Value, A2Value> OutOfDomainPolicy;
///         typedef fern::plus::OutOfRangePolicy<A1Value, A2Value> OutOfRangePolicy;
///         typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
///         typedef fern::algebra::Plus<A1, A2, OutOfDomainPolicy, OutOfRangePolicy,
///             NoDataPolicy> Plus;
/// 
///         Plus plus(NoDataPolicy(result.mask(), true));
/// 
///         plus(argument3, argument1, result);
/// 
///         BOOST_CHECK(!result.mask()[0][0]);
///         BOOST_CHECK_EQUAL(result[0][0], 3);
/// 
///         BOOST_CHECK(!result.mask()[0][1]);
///         BOOST_CHECK_EQUAL(result[0][1], 4);
/// 
///         BOOST_CHECK(!result.mask()[1][0]);
///         BOOST_CHECK_EQUAL(result[1][0], 5);
/// 
///         BOOST_CHECK( result.mask()[1][1]);
/// 
///         BOOST_CHECK(!result.mask()[2][0]);
///         BOOST_CHECK_EQUAL(result[2][0], 6);
/// 
///         BOOST_CHECK(!result.mask()[2][1]);
///         BOOST_CHECK_EQUAL(result[2][1], 7);
///     }
/// }
/// 
/// 
/// struct Task
/// {
/// 
///     // template<class A1, class A2, class R,
///     //     class Indices>
///     template<
///         class Indices>
///     void operator()(
///         fern::Array<int8_t, 2>& argument1,
///         int8_t& argument2,
///         fern::Array<int8_t, 2>& result,
///         Indices indices) const
///     {
///         fern::ArrayView<int8_t, 2> const argument1_view(argument1[indices]);
///         fern::ArrayView<int8_t, 2> result_view(result[indices]);
///         fern::algebra::plus(argument1_view, argument2, result_view);
///     }
/// 
/// };
/// 
/// 
/// BOOST_AUTO_TEST_CASE(threading)
/// {
///     // Create a somewhat larger array.
///     size_t const nr_rows = 6000;
///     size_t const nr_cols = 4000;
///     size_t const stride = 2000;
///     auto extents = fern::extents[nr_rows][nr_cols];
///     fern::Array<int8_t, 2> argument1(extents);
/// 
///     // Fill it with 0, 1, 2, 3, ...
///     std::iota(argument1.data(), argument1.data() + argument1.num_elements(), 0);
/// 
///     int8_t argument2 = 5;
/// 
///     // Create array with values that should be in the result.
///     typedef fern::result<int8_t, int8_t>::type R;
///     fern::Array<R, 2> result_we_want(extents);
///     std::iota(result_we_want.data(), result_we_want.data() +
///         result_we_want.num_elements(), 5);
/// 
/// 
///     // Call plus sequenctially for 6 blocks of stride x stride cells.
///     {
///         fern::Array<R, 2> result_we_got(extents);
///         Task task;
/// 
///         for(size_t r = 0; r < 3; ++r) {
///             for(size_t c = 0; c < 2; ++c) {
///                 size_t row_offset = r * stride;
///                 size_t col_offset = c * stride;
/// 
///                 auto view_indices = fern::indices
///                     [fern::Range(row_offset, row_offset + stride)]
///                     [fern::Range(col_offset, col_offset + stride)];
/// 
///                 task(argument1, argument2, result_we_got, view_indices);
///             }
///         }
/// 
///         // // Verify that the overall result is good.
///         // for(size_t i = 0; i < nr_rows; ++i) {
///         //     for(size_t j = 0; j < nr_cols; ++j) {
///         //         BOOST_CHECK_EQUAL(result_we_got[i][j], result_we_want[i][j]);
///         //     }
///         // }
///     }
/// 
/// 
///     // Call plus concurrently for 6 blocks of stride x stride cells.
///     {
///         fern::Array<R, 2> result_we_got(extents);
///         Task task;
/// 
///         // TODO Fill a task pool with tasks.
/// 
///         // TODO Execute tasks in pool.
/// 
///         std::vector<std::thread> threads;
/// 
///         for(size_t r = 0; r < 3; ++r) {
///             for(size_t c = 0; c < 2; ++c) {
///                 size_t row_offset = r * stride;
///                 size_t col_offset = c * stride;
/// 
///                 auto view_indices = fern::indices
///                     [fern::Range(row_offset, row_offset + stride)]
///                     [fern::Range(col_offset, col_offset + stride)];
/// 
///                 threads.push_back(std::thread(task, std::ref(argument1),
///                     std::ref(argument2), std::ref(result_we_got),
///                     view_indices));
///             }
///         }
/// 
///         for(auto& thread: threads) {
///             thread.join();
///         }
/// 
///         // // Verify that the overall result is good.
///         // for(size_t i = 0; i < nr_rows; ++i) {
///         //     for(size_t j = 0; j < nr_cols; ++j) {
///         //         BOOST_CHECK_EQUAL(result_we_got[i][j], result_we_want[i][j]);
///         //     }
///         // }
///     }
/// 
/// 
///     // TODO Make this work for masked arrays too. The mask policy must contain
///     //      a view too.
/// 
///     // TODO Think about easy ways to select execution scheme. Sequentialy or
///     //      concurrently.
///     //      Refactor this code.
/// 
///     // TODO Make a boolean == which we can use here to verify results. Also
///     //      multi-threaded.
/// }

BOOST_AUTO_TEST_SUITE_END()
