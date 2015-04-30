// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include "fern/language/operation/core/argument.h"


namespace fern {

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

} // namespace fern
