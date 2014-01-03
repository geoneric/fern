#define BOOST_TEST_MODULE fern algorithm policy
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/mark_no_data_by_value.h"


template<
    class T>
struct Mask {

    Mask()
        : _values(nullptr)
    {
    }

    Mask(T* values)
        : _values(values)
    {
    }

    inline void set(
         size_t index,
         T const& value)
    {
         assert(_values);
         _values[index] = value;
    }

    T* _values;

};


template<
    class T,
    class Mask>
struct PolicyHost:
    public fern::MarkNoDataByValue<T, Mask>
{
};


BOOST_AUTO_TEST_SUITE(mark_no_data)

BOOST_AUTO_TEST_CASE(mark_no_data)
{
    PolicyHost<int32_t, Mask<int32_t>> policy;
    int32_t values[] = { 5, 4, 3, 2, 1 };
    Mask<int32_t> mask(values);
    policy.init_no_data_policy(mask, 6);

    policy.mark(0);
    policy.mark(4);
    BOOST_CHECK_EQUAL(values[0], 6);
    BOOST_CHECK_EQUAL(values[1], 4);
    BOOST_CHECK_EQUAL(values[2], 3);
    BOOST_CHECK_EQUAL(values[3], 2);
    BOOST_CHECK_EQUAL(values[4], 6);
}

BOOST_AUTO_TEST_SUITE_END()
