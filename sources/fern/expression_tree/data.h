#pragma once
#include <boost/variant/variant.hpp>
#include "fern/expression_tree/constant.h"
#include "fern/expression_tree/array.h"


namespace fern {

typedef boost::variant<
    Constant<int32_t>,
    Constant<int64_t>,
    Constant<double>,
    Array<int32_t>,
    Array<int64_t>,
    Array<double>
> Data;

} // namespace fern
