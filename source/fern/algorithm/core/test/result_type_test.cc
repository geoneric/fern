// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm core result_type
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/core/result_type.h"
#include "fern/feature/core/data_customization_point/masked_scalar.h"
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/typename.h"
#include "fern/core/data_customization_point/vector.h"


namespace fa = fern::algorithm;


#define verify_result_type(                                       \
    A1, A2, TypeWeWant)                                           \
{                                                                 \
    using TypeWeGet = typename fa::Result<A1, A2>::type;          \
                                                                  \
    BOOST_CHECK_MESSAGE((std::is_same<TypeWeGet, TypeWeWant>()),  \
        fern::typename_<TypeWeGet>() + " != " +                   \
        fern::typename_<TypeWeWant>());                           \
}


BOOST_AUTO_TEST_SUITE(result_type)

BOOST_AUTO_TEST_CASE(result_type)
{
    using namespace fern;

    // Constants.
    verify_result_type(MaskedScalar<int8_t>, MaskedScalar<int8_t>,
        MaskedScalar<int8_t>);
    verify_result_type(int8_t, MaskedScalar<int8_t>, MaskedScalar<int8_t>);
    verify_result_type(MaskedScalar<int8_t>, int8_t, MaskedScalar<int8_t>);

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
