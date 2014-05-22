#pragma once
#include <boost/variant/variant.hpp>
// #include "fern/expression_tree/array.h"
#include "fern/expression_tree/constant.h"
#include "fern/expression_tree/raster.h"


namespace fern {
namespace expression_tree {

using Data = boost::variant<
    Constant<int32_t>,
    Constant<int64_t>,
    Constant<double>,
    // Array<int32_t>,
    // Array<int64_t>,
    // Array<double>,
    Raster<int32_t>,
    Raster<int64_t>,
    Raster<double>
>;

} // expression_tree
} // namespace fern
