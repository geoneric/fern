#define BOOST_TEST_MODULE fern algorithm algebra sum
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/algorithm/algebra/sum.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/core/typename.h"
#include "fern/core/types.h"


#define verify_result_value_type(                                              \
    A1, TypeWeWant)                                                            \
{                                                                              \
    typedef fern::algebra::Sum<A1>::R TypeWeGet;                               \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        fern::typename_<TypeWeGet>() + " != " +                                \
        fern::typename_<TypeWeWant>());                                        \
}


template<
    class A1,
    class R>
void verify_value(
    A1 const& argument1,
    R const& result_we_want)
{
    fern::algebra::Sum<A1> operation;
    R result_we_get;

    operation(argument1, result_we_get);
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


BOOST_AUTO_TEST_SUITE(sum)

BOOST_AUTO_TEST_CASE(result_type)
{
    verify_result_value_type(uint8_t, uint8_t);
    // summing bools is not supported.
    // verify_result_value_type(bool, fern::count_t);
    verify_result_value_type(fern::float64_t, fern::float64_t);

    verify_result_value_type(std::vector<uint8_t>, uint8_t);
    verify_result_value_type(d1::Array<uint8_t>, uint8_t);
    verify_result_value_type(d2::Array<uint8_t>, uint8_t);

    verify_result_value_type(fern::MaskedConstant<uint8_t>,
        fern::MaskedConstant<uint8_t>);
}


BOOST_AUTO_TEST_CASE(constants)
{
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<double, double>(-5.5, -5.5);
    verify_value<double, double>(-5.5, -5.5);
}


BOOST_AUTO_TEST_CASE(masked_constants)
{
    // Masked constant with non-masking sum.
    {
        typedef fern::MaskedConstant<int32_t> A1;
        typedef fern::algebra::Sum<A1> Sum;
        Sum sum;
        A1 argument1(5);

        typedef Sum::R R;

        {
            R result;
            BOOST_CHECK(!result.mask());

            sum(argument1, result);
            BOOST_CHECK(!result.mask());
            BOOST_CHECK_EQUAL(result.value(), 5);
        }

        {
            // Default Sum doesn't check the input nodata mask.
            R result;
            argument1.mask() = true;
            argument1.value() = 6;
            sum(argument1, result);
            BOOST_CHECK(!result.mask());
            BOOST_CHECK_EQUAL(result.value(), 6);
        }
    }

    // Masked constant with masking sum.
    {
        typedef fern::MaskedConstant<int32_t> A1;
        typedef fern::MarkNoDataByValue<bool, bool> NoDataPolicy;
        typedef fern::DiscardRangeErrors OutOfRangePolicy;
        typedef fern::algebra::Sum<A1, OutOfRangePolicy, NoDataPolicy> Sum;
        A1 argument1(5);

        typedef Sum::R R;
        R result;
        result.mask() = argument1.mask();

        Sum sum(NoDataPolicy(result.mask(), true));

        sum(argument1, result);
        BOOST_CHECK(!result.mask());
        BOOST_CHECK_EQUAL(result.value(), 5);

        result.mask() = true;
        argument1.value() = 6;
        BOOST_CHECK(sum.algorithm.is_no_data());
        sum(argument1, result);
        BOOST_CHECK(result.mask());
        BOOST_CHECK_EQUAL(result.value(), 5);
    }
}


BOOST_AUTO_TEST_CASE(collections)
{
    // vector
    {
        std::vector<int32_t> argument1 = { 1, 2, 3, 5 };
        int32_t result;
        fern::algebra::sum(argument1, result);
        BOOST_CHECK_EQUAL(result, 11);
    }

    // 2D array
    {
        size_t const nr_rows = 3;
        size_t const nr_cols = 2;
        auto extents = fern::extents[nr_rows][nr_cols];

        fern::Array<int8_t, 2> argument1(extents);
        argument1[0][0] =  -2;
        argument1[0][1] =  -1;
        argument1[1][0] =  0;
        argument1[1][1] =  9;
        argument1[2][0] =  1;
        argument1[2][1] =  2;

        int8_t result;

        fern::algebra::sum(argument1, result);
        BOOST_CHECK_EQUAL(result, 9);
    }
}


// TODO hier verder
// BOOST_AUTO_TEST_CASE(masked_collections)
// {
//     // 2D masked array
//     {
//         size_t const nr_rows = 3;
//         size_t const nr_cols = 2;
//         auto extents = fern::extents[nr_rows][nr_cols];
// 
//         typedef fern::MaskedConstant<int8_t> A1Value;
//         typedef fern::MaskedArray<A1Value, 2> A1;
// 
//         A1 argument1(extents);
//         argument1[0][0] =  -2;
//         argument1[0][1] =  -1;
//         argument1[1][0] =  0;
//         argument1[1][1] =  9;
//         argument1[2][0] =  1;
//         argument1[2][1] =  2;
// 
//         fern::MaskedConstant<int8_t> result;
// 
//         typedef fern::DiscardRangeErrors OutOfRangePolicy;
//         // typedef fern::MarkNoDataByValue<bool, fern::Mask<2>> NoDataPolicy;
//         // typedef fern::algebra::Sum<A1, OutOfRangePolicy, NoDataPolicy> Sum;
//         // Sum sum(NoDataPolicy(argument1.mask(), true));
// 
//         typedef fern::algebra::Sum<A1, OutOfRangePolicy> Sum;
//         Sum sum;
// 
//         sum(argument1, result);
//         BOOST_CHECK_EQUAL(result, 9);
// 
//         // Mask the 9.
//         argument1.mask()[1][1] =  true;
// 
//         sum(argument1, result);
//         BOOST_CHECK_EQUAL(result, 0);
//     }
// }

BOOST_AUTO_TEST_SUITE_END()
