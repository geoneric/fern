#define BOOST_TEST_MODULE fern algorithm algebra result_type
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/algebra/result_type.h"
#include "fern/feature/core/masked_constant_traits.h"
#include "fern/core/constant_traits.h"
#include "fern/core/typename.h"
#include "fern/core/vector_traits.h"


#define verify_result_type(                                                    \
    A1, A2, TypeWeWant)                                                        \
{                                                                              \
    using TypeWeGet = typename fern::Result<A1, A2>::type;                     \
                                                                               \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),               \
        fern::typename_<TypeWeGet>() + fern::String(" != ") +  \
        fern::typename_<TypeWeWant>()); \
}


BOOST_AUTO_TEST_SUITE(result_type)

BOOST_AUTO_TEST_CASE(result_type)
{
    using namespace fern;

    // Constants.
    verify_result_type(MaskedConstant<int8_t>, MaskedConstant<int8_t>,
        MaskedConstant<int8_t>);
    verify_result_type(int8_t, MaskedConstant<int8_t>, MaskedConstant<int8_t>);
    verify_result_type(MaskedConstant<int8_t>, int8_t, MaskedConstant<int8_t>);

    // Collections.
    verify_result_type(int8_t, std::vector<int8_t>, std::vector<int8_t>);
    verify_result_type(int8_t, std::vector<float>, std::vector<float>);
    verify_result_type(float, std::vector<int8_t>, std::vector<float>);

    verify_result_type(std::vector<int8_t>, int8_t, std::vector<int8_t>);
    verify_result_type(std::vector<float>, int8_t, std::vector<float>);
    verify_result_type(std::vector<int8_t>, float, std::vector<float>);

    verify_result_type(std::vector<int8_t>, std::vector<int8_t>,
        std::vector<int8_t>);
    verify_result_type(std::vector<float>, std::vector<int8_t>,
        std::vector<float>);
    verify_result_type(std::vector<int8_t>, std::vector<float>,
        std::vector<float>);
}

BOOST_AUTO_TEST_SUITE_END()
