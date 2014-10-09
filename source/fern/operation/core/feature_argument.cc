#include "fern/operation/core/feature_argument.h"


namespace fern {

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

} // namespace fern
