#pragma once
#include <memory>
#include "geoneric/operation/core/argument.h"


namespace geoneric {

class Feature;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FeatureArgument:
    public Argument
{

public:

                   FeatureArgument     (
                                  std::shared_ptr<Feature> const& feature);

                   ~FeatureArgument    ()=default;

                   FeatureArgument     (FeatureArgument&&)=delete;

    FeatureArgument& operator=         (FeatureArgument&&)=delete;

                   FeatureArgument     (FeatureArgument const&)=delete;

    FeatureArgument& operator=         (FeatureArgument const&)=delete;

    std::shared_ptr<Feature> const& feature() const;

private:

    std::shared_ptr<Feature> _feature;

};

} // namespace geoneric
