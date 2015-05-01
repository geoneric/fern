// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/core/feature_argument.h"


namespace fern {
namespace language {

FeatureArgument::FeatureArgument(
    std::shared_ptr<Feature> const& feature)

    : Argument(ArgumentType::AT_FEATURE),
      _feature(feature)

{
}


std::shared_ptr<Feature> const& FeatureArgument::feature() const
{
    return _feature;
}

} // namespace language
} // namespace fern
