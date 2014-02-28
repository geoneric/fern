#define BOOST_TEST_MODULE fern algorithm algebra sum
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_traits.h"
#include "fern/core/vector_traits.h"
#include "fern/algorithm/algebra/sum.h"
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
}


BOOST_AUTO_TEST_CASE(constants)
{
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<int8_t, int8_t>(-5, -5);
    verify_value<double, double>(-5.5, -5.5);
    verify_value<double, double>(-5.5, -5.5);
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

BOOST_AUTO_TEST_SUITE_END()
