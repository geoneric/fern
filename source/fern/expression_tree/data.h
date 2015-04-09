// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
