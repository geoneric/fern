#define BOOST_TEST_MODULE fern algorithm algebra equal
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_traits.h"
#include "fern/core/constant_traits.h"
#include "fern/core/vector_traits.h"
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

    // array == masked_array
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

    // array == 5
    {
        fern::Array<bool, 2> result(extents);
        fern::algebra::equal(argument1, argument3, result);

        BOOST_CHECK(!result[0][0]);
        BOOST_CHECK(!result[0][1]);
        BOOST_CHECK(!result[1][0]);
        BOOST_CHECK(!result[1][1]);
        BOOST_CHECK( result[2][0]);
        BOOST_CHECK( result[2][1]);
    }

    // 5 == array
    {
        fern::Array<bool, 2> result(extents);
        fern::algebra::equal(argument3, argument1, result);

        BOOST_CHECK(!result[0][0]);
        BOOST_CHECK(!result[0][1]);
        BOOST_CHECK(!result[1][0]);
        BOOST_CHECK(!result[1][1]);
        BOOST_CHECK( result[2][0]);
        BOOST_CHECK( result[2][1]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
