// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm trigonometry atan
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/trigonometry/atan.h"


namespace fa = fern::algorithm;


template<
    class Value,
    class Result>
void verify_value(
    Value const& value,
    Result const& result_we_want)
{
    fa::SequentialExecutionPolicy sequential;

    Result result_we_get;
    fa::trigonometry::atan(sequential, value, result_we_get);
    BOOST_CHECK_CLOSE(result_we_get, result_we_want, 1e-5);
}


BOOST_AUTO_TEST_CASE(algorithm)
{
    verify_value<float, float>(0.0f, 0.0f);
    verify_value<float, float>(-0.0f, -0.0f);
    verify_value<float, float>(fern::infinity<float>(),
        fern::pi<float>() / 2.0f);
    verify_value<float, float>(-fern::infinity<float>(),
        -fern::pi<float>() / 2.0f);
}
