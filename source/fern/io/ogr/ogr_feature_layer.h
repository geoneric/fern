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
#include "fern/core/string.h"


class OGRLayer;

namespace fern {

class Domain;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRFeatureLayer
{

    friend class OGRFeatureLayerTest;

public:

                   OGRFeatureLayer     (OGRFeatureLayer const&)=delete;

    OGRFeatureLayer& operator=         (OGRFeatureLayer const&)=delete;

                   OGRFeatureLayer     (OGRLayer* const layer);

                   ~OGRFeatureLayer    ();

    String         name                () const;

    Domain const&  domain              () const;

    template<class Domain>
    std::shared_ptr<Domain> domain     () const;

private:

    OGRLayer* const  _layer;

    std::shared_ptr<Domain> _domain;

};


template<class Domain>
std::shared_ptr<Domain> OGRFeatureLayer::domain() const
{
    return std::dynamic_pointer_cast<Domain>(_domain);
}

} // namespace fern
